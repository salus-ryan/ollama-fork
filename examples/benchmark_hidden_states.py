#!/usr/bin/env python3
"""
Benchmarking framework for hidden states capture performance analysis.
Measures latency/token, memory overhead, and compression trade-offs.
"""

import time
import json
import psutil
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import threading
import concurrent.futures
from collections import defaultdict
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
try:
    from nltk.translate.bleu_score import sentence_bleu
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    tokens_per_second: float
    memory_peak_mb: float
    memory_overhead_mb: float
    compression_ratio: float
    quality_score: Optional[float] = None
    hidden_state_size_mb: Optional[float] = None
    vram_peak_mb: Optional[float] = None
    vram_overhead_mb: Optional[float] = None
    tensor_elements: Optional[int] = None
    actual_tensor_bytes: Optional[int] = None

class HiddenStatesBenchmark:
    """Comprehensive benchmarking suite for hidden states performance."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        import os
        # Allow overriding via OLLAMA_HOST (e.g., 127.0.0.1:11501)
        host = os.getenv("OLLAMA_HOST")
        if host:
            if not host.startswith("http"):
                host = f"http://{host}"
            self.base_url = host
        else:
            self.base_url = base_url
        self.process = psutil.Process()
        self.cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
        
    def get_gpu_memory(self) -> Tuple[float, float]:
        """Get GPU memory usage (used, total) in MB."""
        if self.cuda_available:
            try:
                mem_info = torch.cuda.mem_get_info()
                free_mb = mem_info[0] / 1024 / 1024
                total_mb = mem_info[1] / 1024 / 1024
                used_mb = total_mb - free_mb
                return used_mb, total_mb
            except Exception:
                pass
        
        # Fallback to nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                used, total = map(int, lines[0].split(', '))
                return float(used), float(total)
        except Exception:
            pass
        
        return 0.0, 0.0
        
    def measure_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_generation(
        self, 
        model: str,
        prompt: str,
        hidden_config: Optional[Dict] = None,
        num_tokens: int = 100
    ) -> BenchmarkResult:
        """Benchmark a single generation run with optional hidden states."""
        
        # Measure baseline memory (both RAM and VRAM)
        baseline_memory = self.measure_memory_usage()
        baseline_vram, _ = self.get_gpu_memory()
        
        # Prepare request
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": num_tokens}
        }
        
        if hidden_config:
            # Server expects the field name 'expose_hidden' in the request body
            # Hidden state capture is disabled unless `enabled` is explicitly true.
            cfg = {"enabled": True}
            cfg.update(hidden_config)
            request_data["expose_hidden"] = cfg
        
        # Start timing
        start_time = time.time()
        peak_memory = baseline_memory
        peak_vram = baseline_vram
        tokens_generated = 0
        hidden_state_size = 0
        tensor_elements = 0
        generated_text = ""
        hidden_states_captured = False
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                stream=True
            )
            
            if response.status_code != 200:
                print(f"HTTP Error {response.status_code}: {response.text}")
                return BenchmarkResult(0, 0, 0, 0)
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # Track memory usage (RAM and VRAM)
                    current_memory = self.measure_memory_usage()
                    current_vram, _ = self.get_gpu_memory()
                    peak_memory = max(peak_memory, current_memory)
                    peak_vram = max(peak_vram, current_vram)
                    
                    # Count tokens and collect text for quality analysis
                    if 'response' in data and data['response']:
                        tokens_generated += 1
                        generated_text += data['response']
                    
                    # Measure hidden state size properly
                    if 'hidden_states' in data and data['hidden_states']:
                        hidden_states_captured = True
                        hidden_states = data['hidden_states']
                        
                        if isinstance(hidden_states, list):
                            # Handle list format
                            for layer_data in hidden_states:
                                if isinstance(layer_data, dict) and 'data' in layer_data:
                                    try:
                                        arr = np.asarray(layer_data['data'], dtype=np.float32)
                                        tensor_elements += arr.size
                                        
                                        # Calculate size based on compression
                                        if hidden_config and 'compression' in hidden_config:
                                            if hidden_config['compression'] == 'float16':
                                                hidden_state_size += arr.astype(np.float16).nbytes
                                            elif hidden_config['compression'] == 'int8':
                                                hidden_state_size += arr.astype(np.int8).nbytes
                                            else:
                                                hidden_state_size += arr.nbytes
                                        else:
                                            hidden_state_size += arr.nbytes
                                    except Exception as e:
                                        print(f"Error processing hidden state data: {e}")
                                        hidden_state_size += len(str(layer_data))
                        
                        elif isinstance(hidden_states, dict):
                            # Handle dict format
                            for layer_name, layer_data in hidden_states.items():
                                if isinstance(layer_data, list):
                                    try:
                                        arr = np.asarray(layer_data, dtype=np.float32)
                                        tensor_elements += arr.size
                                        
                                        # Calculate size based on compression
                                        if hidden_config and 'compression' in hidden_config:
                                            if hidden_config['compression'] == 'float16':
                                                hidden_state_size += arr.astype(np.float16).nbytes
                                            elif hidden_config['compression'] == 'int8':
                                                hidden_state_size += arr.astype(np.int8).nbytes
                                            else:
                                                hidden_state_size += arr.nbytes
                                        else:
                                            hidden_state_size += arr.nbytes
                                    except Exception as e:
                                        print(f"Error processing layer {layer_name}: {e}")
                                        hidden_state_size += len(str(layer_data))
                    
                    if data.get('done', False):
                        break
        
        except Exception as e:
            print(f"Benchmark error: {e}")
            return BenchmarkResult(0, 0, 0, 0)
        
        # If hidden states were requested but not captured, record this fact
        if hidden_config and not hidden_states_captured:
            print(f"Warning: Hidden states requested but not received from server.")
            # Don't simulate - leave as 0 to show actual server behavior
        
        # Calculate metrics
        end_time = time.time()
        duration = end_time - start_time
        tokens_per_second = tokens_generated / duration if duration > 0 else 0
        memory_overhead = peak_memory - baseline_memory
        vram_overhead = peak_vram - baseline_vram
        hidden_state_size_mb = hidden_state_size / (1024 * 1024) if hidden_state_size > 0 else 0
        
        return BenchmarkResult(
            tokens_per_second=tokens_per_second,
            memory_peak_mb=peak_memory,
            memory_overhead_mb=memory_overhead,
            compression_ratio=1.0,  # Will be calculated separately
            hidden_state_size_mb=hidden_state_size_mb,
            vram_peak_mb=peak_vram,
            vram_overhead_mb=vram_overhead,
            tensor_elements=tensor_elements,
            actual_tensor_bytes=hidden_state_size
        )
    
    def calculate_quality_score(self, baseline_text: str, hidden_text: str) -> float:
        """Calculate quality score comparing baseline vs hidden state output."""
        if not NLTK_AVAILABLE or not baseline_text or not hidden_text:
            return 0.0
        
        try:
            # Simple BLEU score calculation
            baseline_tokens = baseline_text.lower().split()
            hidden_tokens = hidden_text.lower().split()
            
            if len(baseline_tokens) == 0 or len(hidden_tokens) == 0:
                return 0.0
            
            bleu = sentence_bleu([baseline_tokens], hidden_tokens)
            return bleu
        except Exception:
            return 0.0
    
    def benchmark_compression_modes(
        self, 
        model: str, 
        prompt: str,
        base_config: Dict = None
    ) -> Dict[str, BenchmarkResult]:
        """Benchmark different compression modes."""
        
        compression_modes = [
            {"compression": "none"},
            {"compression": "float16"},
            {"compression": "int8"}
        ]
        
        # Merge base_config (e.g., last_layer_only) into each compression mode
        if base_config:
            for mode in compression_modes:
                merged = {}
                merged.update(base_config)
                # remove helper flag if present
                if merged.get("all_layers"):
                    merged.pop("all_layers", None)
                mode.update(merged)
        
        results = {}
        
        # Baseline without hidden states
        baseline = self.benchmark_generation(model, prompt, None)
        results["baseline"] = baseline
        
        # Store uncompressed size for ratio calculations
        uncompressed_size = None
        baseline_text = ""
        
        # Test each compression mode
        for mode in compression_modes:
            mode_name = mode["compression"]
            print(f"Benchmarking {mode_name} compression...")
            
            result = self.benchmark_generation(model, prompt, mode)
            
            # Store uncompressed size from first run
            if mode_name == "none":
                uncompressed_size = result.hidden_state_size_mb
                # Generate baseline text for quality comparison
                baseline_result = self.benchmark_generation(model, prompt, None)
            
            # Calculate compression ratio relative to uncompressed
            if uncompressed_size and uncompressed_size > 0 and result.hidden_state_size_mb > 0:
                result.compression_ratio = uncompressed_size / result.hidden_state_size_mb
            
            results[mode_name] = result
        
        return results
    
    def benchmark_concurrency(
        self,
        model: str,
        prompt: str,
        num_concurrent: int = 4,
        hidden_config: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Benchmark concurrent request performance."""
        
        def single_request():
            return self.benchmark_generation(model, prompt, hidden_config, num_tokens=50)
        
        # Measure sequential baseline
        start_time = time.time()
        sequential_results = []
        for _ in range(num_concurrent):
            result = single_request()
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Measure concurrent performance
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            concurrent_futures = [executor.submit(single_request) for _ in range(num_concurrent)]
            concurrent_results = [f.result() for f in concurrent_futures]
        concurrent_time = time.time() - start_time
        
        # Calculate metrics
        sequential_tokens_per_sec = sum(r.tokens_per_second for r in sequential_results)
        concurrent_tokens_per_sec = sum(r.tokens_per_second for r in concurrent_results)
        
        return {
            "sequential_time": sequential_time,
            "concurrent_time": concurrent_time,
            "speedup_ratio": sequential_time / concurrent_time if concurrent_time > 0 else 0,
            "sequential_throughput": sequential_tokens_per_sec,
            "concurrent_throughput": concurrent_tokens_per_sec,
            "throughput_scaling": concurrent_tokens_per_sec / sequential_tokens_per_sec if sequential_tokens_per_sec > 0 else 0
        }
    
    def run_comprehensive_benchmark(
        self, 
        model: str,
        test_prompts: List[str],
        output_dir: str = "benchmark_results"
    ) -> Dict:
        """Run comprehensive benchmark suite."""
        
        Path(output_dir).mkdir(exist_ok=True)
        
        all_results = {
            "model": model,
            "timestamp": time.time(),
            "prompts": {},
            "summary": {}
        }
        
        print(f"Running comprehensive benchmark for {model}")
        
        for i, prompt in enumerate(test_prompts):
            print(f"\nTesting prompt {i+1}/{len(test_prompts)}: {prompt[:50]}...")
            
            # Test different capture configurations
            capture_configs = [
                None,  # No hidden states
                {"last_layer_only": True},  # Last layer only
                {"all_layers": True},  # All layers (no specific layer selection)
            ]
            
            prompt_results = {}
            
            # Add concurrency test
            print(f"  Testing concurrency performance...")
            concurrency_results = self.benchmark_concurrency(
                model, prompt, num_concurrent=4, hidden_config={"last_layer_only": True, "compression": "float16"}
            )
            prompt_results["concurrency"] = concurrency_results
            
            for j, cfg in enumerate(capture_configs):
                config_name = f"config_{j}"
                if cfg is None:
                    config_name = "no_hidden"
                elif cfg.get("last_layer_only"):
                    config_name = "last_layer"
                elif cfg.get("all_layers"):
                    config_name = "all_layers"
                
                print(f"  Testing {config_name}...")
                
                if cfg is None:
                    result = self.benchmark_generation(model, prompt, None)
                else:
                    compression_results = self.benchmark_compression_modes(
                        model, prompt, cfg
                    )
                    result = compression_results
                
                prompt_results[config_name] = result
            
            all_results["prompts"][f"prompt_{i}"] = prompt_results
        
        # Calculate summary statistics
        self._calculate_summary_stats(all_results)
        
        # Save results
        results_file = Path(output_dir) / f"benchmark_{model}_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Generate plots
        self._generate_plots(all_results, output_dir)
        
        print(f"\nBenchmark complete! Results saved to {results_file}")
        print(f"Plots saved to {Path(output_dir) / 'benchmark_analysis.png'}")
        
        return all_results
    
    def _calculate_summary_stats(self, results: Dict):
        """Calculate summary statistics across all runs."""
        
        # Collect metrics across all runs
        tokens_per_second = []
        memory_overhead = []
        hidden_state_sizes = []
        
        for prompt_results in results["prompts"].values():
            for config_results in prompt_results.values():
                if isinstance(config_results, dict) and "baseline" in config_results:
                    # Compression mode results
                    for mode_result in config_results.values():
                        if isinstance(mode_result, BenchmarkResult):
                            tokens_per_second.append(mode_result.tokens_per_second)
                            memory_overhead.append(mode_result.memory_overhead_mb)
                            if mode_result.hidden_state_size_mb:
                                hidden_state_sizes.append(mode_result.hidden_state_size_mb)
                elif isinstance(config_results, BenchmarkResult):
                    # Single result
                    tokens_per_second.append(config_results.tokens_per_second)
                    memory_overhead.append(config_results.memory_overhead_mb)
                    if config_results.hidden_state_size_mb:
                        hidden_state_sizes.append(config_results.hidden_state_size_mb)
        
        results["summary"] = {
            "avg_tokens_per_second": np.mean(tokens_per_second) if tokens_per_second else 0,
            "avg_memory_overhead_mb": np.mean(memory_overhead) if memory_overhead else 0,
            "avg_hidden_state_size_mb": np.mean(hidden_state_sizes) if hidden_state_sizes else 0,
            "total_runs": len(tokens_per_second)
        }
    
    def _generate_plots(self, results: Dict, output_dir: str):
        """Generate visualization plots for benchmark results."""
        
        # Performance vs Memory Trade-off Plot
        plt.figure(figsize=(16, 12))
        
        # Extract data for plotting
        configs = []
        tokens_per_sec = []
        memory_usage = []
        hidden_sizes = []
        
        for prompt_key, prompt_results in results["prompts"].items():
            for config_name, config_results in prompt_results.items():
                if isinstance(config_results, dict) and "baseline" in config_results:
                    for mode_name, mode_result in config_results.items():
                        if isinstance(mode_result, BenchmarkResult):
                            configs.append(f"{config_name}_{mode_name}")
                            tokens_per_sec.append(mode_result.tokens_per_second)
                            memory_usage.append(mode_result.memory_overhead_mb)
                            hidden_sizes.append(mode_result.hidden_state_size_mb or 0)
        
        # Only plot if we have actual data
        if not configs or not tokens_per_sec:
            print("No benchmark data available to plot.")
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        
        # Tokens per second comparison
        bars1 = ax1.bar(range(len(configs)), tokens_per_sec, color='skyblue', alpha=0.7)
        ax1.set_title('Tokens per Second by Configuration', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Tokens/sec', fontsize=12)
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, tokens_per_sec):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Memory overhead comparison
        bars2 = ax2.bar(range(len(configs)), memory_usage, color='lightcoral', alpha=0.7)
        ax2.set_title('Memory Overhead by Configuration', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Memory (MB)', fontsize=12)
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, memory_usage):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)
        
        # Hidden state sizes
        bars3 = ax3.bar(range(len(configs)), hidden_sizes, color='lightgreen', alpha=0.7)
        ax3.set_title('Hidden State Size by Configuration', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Size (MB)', fontsize=12)
        ax3.set_xticks(range(len(configs)))
        ax3.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars3, hidden_sizes):
            if value > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{value:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Performance vs Memory scatter
        scatter = ax4.scatter(memory_usage, tokens_per_sec, s=100, alpha=0.7, c=hidden_sizes, 
                            cmap='viridis', edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Memory Overhead (MB)', fontsize=12)
        ax4.set_ylabel('Tokens per Second', fontsize=12)
        ax4.set_title('Performance vs Memory Trade-off', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar for hidden state sizes
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Hidden State Size (MB)', fontsize=10)
        
        # VRAM usage comparison (if available)
        vram_usage = []
        for prompt_key, prompt_results in results["prompts"].items():
            for config_name, config_results in prompt_results.items():
                if isinstance(config_results, dict) and "baseline" in config_results:
                    for mode_name, mode_result in config_results.items():
                        if isinstance(mode_result, BenchmarkResult) and mode_result.vram_overhead_mb:
                            vram_usage.append(mode_result.vram_overhead_mb)
        
        if vram_usage:
            ax5.bar(range(len(vram_usage)), vram_usage)
            ax5.set_title('VRAM Overhead by Configuration')
            ax5.set_ylabel('VRAM (MB)')
            ax5.tick_params(axis='x', rotation=45)
        else:
            ax5.text(0.5, 0.5, 'No VRAM data available', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('VRAM Usage (Not Available)')
        
        # Concurrency performance
        concurrency_data = []
        for prompt_key, prompt_results in results["prompts"].items():
            if "concurrency" in prompt_results:
                conc_result = prompt_results["concurrency"]
                concurrency_data.append(conc_result.get("throughput_scaling", 0))
        
        if concurrency_data:
            ax6.bar(range(len(concurrency_data)), concurrency_data)
            ax6.set_title('Concurrency Throughput Scaling')
            ax6.set_ylabel('Scaling Factor')
            ax6.axhline(y=1.0, color='r', linestyle='--', label='Linear scaling')
            ax6.legend()
        else:
            ax6.text(0.5, 0.5, 'No concurrency data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Concurrency Scaling (Not Available)')
        
        plt.tight_layout(pad=3.0)
        output_path = Path(output_dir) / 'benchmark_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Generated visualization: {output_path}")
        print(f"Plot contains {len(configs)} configurations with meaningful data")

def main():
    """Main benchmark execution."""
    
    benchmark = HiddenStatesBenchmark()
    
    # Check system capabilities
    print("System Capabilities:")
    print(f"  CUDA Available: {benchmark.cuda_available}")
    print(f"  NLTK Available: {NLTK_AVAILABLE}")
    print(f"  PyTorch Available: {TORCH_AVAILABLE}")
    
    if benchmark.cuda_available:
        vram_used, vram_total = benchmark.get_gpu_memory()
        print(f"  GPU Memory: {vram_used:.1f}/{vram_total:.1f} MB")
    
    print()
    
    # Test prompts for different scenarios
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot learning to paint.",
        "Solve this math problem: What is the derivative of x^3 + 2x^2 - 5x + 1?",
        "Describe the process of photosynthesis step by step.",
        "Generate a Python function to calculate fibonacci numbers."
    ]
    
    # Run comprehensive benchmark
    try:
        import os
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        results = benchmark.run_comprehensive_benchmark(
            model=model_name,  # Use available model or override via OLLAMA_MODEL
            test_prompts=test_prompts
        )
    except Exception as e:
        print(f"Benchmark failed: {e}")
        print("Make sure Ollama is running and the model is available.")
        return
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"Model: {results['model']}")
    print(f"Total runs: {results['summary']['total_runs']}")
    print(f"Avg tokens/sec: {results['summary']['avg_tokens_per_second']:.2f}")
    print(f"Avg memory overhead: {results['summary']['avg_memory_overhead_mb']:.2f} MB")
    print(f"Avg hidden state size: {results['summary']['avg_hidden_state_size_mb']:.2f} MB")
    
    # Print concurrency results if available
    for prompt_key, prompt_results in results["prompts"].items():
        if "concurrency" in prompt_results:
            conc = prompt_results["concurrency"]
            print(f"\nConcurrency Results ({prompt_key}):")
            print(f"  Speedup ratio: {conc.get('speedup_ratio', 0):.2f}x")
            print(f"  Throughput scaling: {conc.get('throughput_scaling', 0):.2f}x")
            break

if __name__ == "__main__":
    main()
