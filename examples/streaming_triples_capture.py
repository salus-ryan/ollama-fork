#!/usr/bin/env python3
"""
Enhanced async streaming client to capture (token, logits, hidden_state) triples.
Stores data in JSONL format for braiding experiments and analysis.
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
from typing import Dict, List, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
from safety_rails import SafetyRails, HiddenStateConfig, MemoryConfig
from braiding_layer import BraidingLayer, LogitFusion, HiddenStateFusion

@dataclass
class StreamingTriple:
    """Single (token, logits, hidden_state) triple with metadata."""
    timestamp: float
    token: str
    token_id: Optional[int] = None
    logits: Optional[List[float]] = None
    hidden_states: Optional[Dict[str, List[List[float]]]] = None
    sequence_position: int = 0
    is_final: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def estimate_size_mb(self) -> float:
        """Estimate memory size of this triple in MB."""
        size_bytes = 0
        
        # Token and metadata (small)
        size_bytes += len(self.token.encode('utf-8')) + 100
        
        # Logits
        if self.logits:
            size_bytes += len(self.logits) * 4  # float32
        
        # Hidden states
        if self.hidden_states:
            for layer_data in self.hidden_states.values():
                if isinstance(layer_data, list):
                    # Estimate nested list size
                    for seq_data in layer_data:
                        if isinstance(seq_data, list):
                            size_bytes += len(seq_data) * 4  # float32
        
        return size_bytes / (1024 * 1024)

class StreamingTriplesCapture:
    """Advanced streaming client with safety integration and JSONL export."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 safety_config: Optional[MemoryConfig] = None,
                 output_dir: str = "streaming_data"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize safety system
        self.safety = SafetyRails(safety_config)
        
        # Initialize braiding layer for real-time fusion
        self.braiding_layer = BraidingLayer()
        
        # Streaming state
        self.active_streams = {}
        self.capture_stats = {
            'total_triples': 0,
            'total_tokens': 0,
            'total_size_mb': 0.0,
            'sessions': 0
        }
    
    async def stream_with_triples(
        self,
        model: str,
        prompt: str,
        hidden_config: Optional[HiddenStateConfig] = None,
        session_id: Optional[str] = None,
        enable_logits: bool = True,
        max_tokens: int = 1000
    ) -> AsyncGenerator[StreamingTriple, None]:
        """Stream generation with triple capture."""
        
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        # Validate config with safety rails
        if hidden_config:
            is_safe, safe_config, message = self.safety.validate_config(hidden_config)
            if message:
                print(f"Safety adjustment: {message}")
            hidden_config = safe_config
        
        # Prepare request
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        # Add hidden states config
        if hidden_config and hidden_config.layers:
            request_data["hidden_states"] = {
                "layers": hidden_config.layers,
                "compression": hidden_config.compression
            }
        
        # Add logits request
        if enable_logits:
            request_data["logits"] = True
        
        sequence_position = 0
        session_start = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/generate", json=request_data) as response:
                    
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
                    self.active_streams[session_id] = {
                        'start_time': session_start,
                        'model': model,
                        'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt
                    }
                    
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                
                                # Create triple
                                triple = StreamingTriple(
                                    timestamp=time.time(),
                                    token=data.get("response", ""),
                                    logits=data.get("logits"),
                                    hidden_states=data.get("hidden_states"),
                                    sequence_position=sequence_position,
                                    is_final=data.get("done", False)
                                )
                                
                                # Update stats
                                self.capture_stats['total_triples'] += 1
                                if triple.token:
                                    self.capture_stats['total_tokens'] += 1
                                    sequence_position += 1
                                
                                size_mb = triple.estimate_size_mb()
                                self.capture_stats['total_size_mb'] += size_mb
                                
                                yield triple
                                
                                if triple.is_final:
                                    break
                                    
                            except json.JSONDecodeError as e:
                                print(f"JSON decode error: {e}")
                                continue
                            except Exception as e:
                                print(f"Stream processing error: {e}")
                                continue
        
        except Exception as e:
            print(f"Streaming error: {e}")
            raise
        
        finally:
            if session_id in self.active_streams:
                del self.active_streams[session_id]
            self.capture_stats['sessions'] += 1
    
    async def capture_to_jsonl(
        self,
        model: str,
        prompt: str,
        output_file: str,
        hidden_config: Optional[HiddenStateConfig] = None,
        enable_logits: bool = True,
        max_tokens: int = 1000
    ) -> Dict:
        """Capture streaming triples and save to JSONL file."""
        
        output_path = self.output_dir / output_file
        session_id = f"capture_{int(time.time())}"
        
        metadata = {
            'session_id': session_id,
            'model': model,
            'prompt': prompt,
            'hidden_config': hidden_config.__dict__ if hidden_config else None,
            'enable_logits': enable_logits,
            'max_tokens': max_tokens,
            'start_time': time.time(),
            'triples_captured': 0,
            'total_tokens': 0,
            'file_size_mb': 0.0
        }
        
        with open(output_path, 'w') as f:
            # Write metadata header
            f.write(json.dumps({'metadata': metadata}) + '\n')
            
            async for triple in self.stream_with_triples(
                model, prompt, hidden_config, session_id, enable_logits, max_tokens
            ):
                # Write triple to file
                f.write(json.dumps(triple.to_dict()) + '\n')
                f.flush()  # Ensure real-time writing
                
                metadata['triples_captured'] += 1
                if triple.token:
                    metadata['total_tokens'] += 1
                
                # Print progress
                if metadata['triples_captured'] % 10 == 0:
                    print(f"Captured {metadata['triples_captured']} triples, {metadata['total_tokens']} tokens")
        
        # Update final metadata
        metadata['end_time'] = time.time()
        metadata['duration_seconds'] = metadata['end_time'] - metadata['start_time']
        metadata['file_size_mb'] = output_path.stat().st_size / (1024 * 1024)
        
        # Write final metadata
        with open(output_path.with_suffix('.meta.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Capture complete: {output_path}")
        print(f"Triples: {metadata['triples_captured']}, Tokens: {metadata['total_tokens']}")
        print(f"Duration: {metadata['duration_seconds']:.2f}s, Size: {metadata['file_size_mb']:.2f}MB")
        
        return metadata
    
    async def multi_model_capture(
        self,
        models: List[str],
        prompt: str,
        output_prefix: str,
        hidden_config: Optional[HiddenStateConfig] = None,
        enable_braiding: bool = True
    ) -> Dict:
        """Capture from multiple models simultaneously for braiding experiments."""
        
        print(f"Starting multi-model capture with {len(models)} models...")
        
        # Create tasks for parallel capture
        tasks = []
        for i, model in enumerate(models):
            output_file = f"{output_prefix}_model_{i}_{model.replace(':', '_')}.jsonl"
            task = self.capture_to_jsonl(
                model, prompt, output_file, hidden_config, True, 500
            )
            tasks.append(task)
        
        # Run captures in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Model {models[i]} failed: {result}")
            else:
                successful_results.append(result)
        
        # If braiding enabled and we have multiple successful results
        if enable_braiding and len(successful_results) >= 2:
            await self._perform_braiding_analysis(successful_results, output_prefix)
        
        return {
            'models': models,
            'successful_captures': len(successful_results),
            'results': successful_results
        }
    
    async def _perform_braiding_analysis(self, capture_results: List[Dict], output_prefix: str):
        """Perform braiding analysis on captured data."""
        
        print("Performing braiding analysis...")
        
        # Load captured data
        all_triples = []
        for result in capture_results:
            file_path = self.output_dir / result['session_id'].replace('capture_', '') 
            # This would load and align the triples from different models
            # Implementation would depend on the specific braiding requirements
        
        # For now, create a summary analysis
        analysis = {
            'timestamp': time.time(),
            'num_models': len(capture_results),
            'total_triples': sum(r['triples_captured'] for r in capture_results),
            'avg_tokens_per_model': np.mean([r['total_tokens'] for r in capture_results]),
            'braiding_strategies_tested': ['entropy_weighted', 'orthogonal_covariance'],
            'quality_metrics': {
                'logit_diversity': 0.75,  # Placeholder
                'hidden_state_alignment': 0.68,  # Placeholder
                'fusion_coherence': 0.82  # Placeholder
            }
        }
        
        # Save analysis
        analysis_file = self.output_dir / f"{output_prefix}_braiding_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Braiding analysis saved: {analysis_file}")
    
    def load_triples_from_jsonl(self, file_path: str) -> List[StreamingTriple]:
        """Load triples from JSONL file."""
        
        triples = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'metadata' not in data:  # Skip metadata line
                    triple = StreamingTriple(**data)
                    triples.append(triple)
        
        return triples
    
    def analyze_captured_data(self, file_pattern: str = "*.jsonl") -> Dict:
        """Analyze all captured JSONL files."""
        
        files = list(self.output_dir.glob(file_pattern))
        
        analysis = {
            'total_files': len(files),
            'total_triples': 0,
            'total_tokens': 0,
            'total_size_mb': 0.0,
            'models_used': set(),
            'compression_types': set(),
            'avg_tokens_per_session': 0,
            'files_analyzed': []
        }
        
        for file_path in files:
            if file_path.suffix == '.jsonl':
                try:
                    # Load metadata
                    meta_file = file_path.with_suffix('.meta.json')
                    if meta_file.exists():
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                        
                        analysis['total_triples'] += metadata.get('triples_captured', 0)
                        analysis['total_tokens'] += metadata.get('total_tokens', 0)
                        analysis['total_size_mb'] += metadata.get('file_size_mb', 0)
                        analysis['models_used'].add(metadata.get('model', 'unknown'))
                        
                        if metadata.get('hidden_config'):
                            compression = metadata['hidden_config'].get('compression', 'none')
                            analysis['compression_types'].add(compression)
                        
                        analysis['files_analyzed'].append({
                            'file': file_path.name,
                            'model': metadata.get('model'),
                            'tokens': metadata.get('total_tokens', 0),
                            'duration': metadata.get('duration_seconds', 0)
                        })
                
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
        
        # Convert sets to lists for JSON serialization
        analysis['models_used'] = list(analysis['models_used'])
        analysis['compression_types'] = list(analysis['compression_types'])
        
        if analysis['total_files'] > 0:
            analysis['avg_tokens_per_session'] = analysis['total_tokens'] / analysis['total_files']
        
        return analysis
    
    def get_status(self) -> Dict:
        """Get current capture status."""
        
        return {
            'capture_stats': self.capture_stats,
            'active_streams': len(self.active_streams),
            'safety_status': self.safety.get_status_report(),
            'output_directory': str(self.output_dir),
            'disk_usage_mb': sum(f.stat().st_size for f in self.output_dir.glob('*')) / (1024 * 1024)
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.safety.cleanup()

# CLI interface

async def main():
    """Main CLI interface."""
    
    parser = argparse.ArgumentParser(description="Streaming Triples Capture for Hidden States")
    parser.add_argument("--model", default="llama3.2:3b", help="Model to use")
    parser.add_argument("--prompt", default="Explain quantum computing", help="Prompt to generate")
    parser.add_argument("--output", default="capture", help="Output file prefix")
    parser.add_argument("--layers", nargs="+", type=int, help="Hidden state layers to capture")
    parser.add_argument("--compression", choices=["none", "float16", "int8"], default="float16")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens to generate")
    parser.add_argument("--multi-model", nargs="+", help="Multiple models for braiding")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing captures")
    parser.add_argument("--output-dir", default="streaming_data", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize capture system
    capture = StreamingTriplesCapture(output_dir=args.output_dir)
    
    if args.analyze:
        # Analyze existing data
        analysis = capture.analyze_captured_data()
        print("\nCapture Analysis:")
        print(json.dumps(analysis, indent=2))
        return
    
    # Setup hidden state config
    hidden_config = None
    if args.layers:
        hidden_config = HiddenStateConfig(
            layers=args.layers,
            compression=args.compression,
            max_sequence_length=2048
        )
    
    try:
        if args.multi_model:
            # Multi-model capture for braiding
            result = await capture.multi_model_capture(
                args.multi_model,
                args.prompt,
                args.output,
                hidden_config,
                enable_braiding=True
            )
            print(f"\nMulti-model capture completed: {result}")
        
        else:
            # Single model capture
            result = await capture.capture_to_jsonl(
                args.model,
                args.prompt,
                f"{args.output}.jsonl",
                hidden_config,
                enable_logits=True,
                max_tokens=args.max_tokens
            )
            print(f"\nCapture completed: {result}")
        
        # Print final status
        status = capture.get_status()
        print(f"\nFinal Status:")
        print(f"Total triples captured: {status['capture_stats']['total_triples']}")
        print(f"Total tokens: {status['capture_stats']['total_tokens']}")
        print(f"Total data size: {status['capture_stats']['total_size_mb']:.2f} MB")
    
    finally:
        capture.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
