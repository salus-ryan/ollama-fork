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
            request_data["hidden_states"] = hidden_config
        
        # Start timing
        start_time = time.time()
        peak_memory = baseline_memory
        peak_vram = baseline_vram
        tokens_generated = 0
        hidden_state_size = 0
        tensor_elements = 0
        generated_text = ""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    
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
                    if 'hidden_states' in data:
                        hidden_states = data['hidden_states']
                        if isinstance(hidden_states, dict):
                            for layer_name, layer_data in hidden_states.items():
                                if isinstance(layer_data, list):
                                    # Calculate actual tensor size
                                    try:
                                        arr = np.asarray(layer_data, dtype=np.float32)
                                        tensor_elements += arr.size
                                        
                                        # Estimate compression sizes
                                        if hidden_config and 'compression' in hidden_config:
                                            if hidden_config['compression'] == 'float16':
                                                hidden_state_size += arr.astype(np.float16).nbytes
                                            elif hidden_config['compression'] == 'int8':
                                                hidden_state_size += arr.astype(np.int8).nbytes
                                            else:
                                                hidden_state_size += arr.nbytes
                                        else:
                                            hidden_state_size += arr.nbytes
                                    except Exception:
                                        # Fallback to string length
                                        hidden_state_size += len(str(layer_data))
                    
                    if data.get('done', False):
                        break
        
        except Exception as e:
            print(f"Benchmark error: {e}")
            return BenchmarkResult(0, 0, 0, 0)
        
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
        layers: List[int] = None
    ) -> Dict[str, BenchmarkResult]:
        """Benchmark different compression modes."""
        
        compression_modes = [
            {"compression": "none"},
            {"compression": "float16"},
            {"compression": "int8"}
        ]
        
        if layers:
            for mode in compression_modes:
                mode["layers"] = layers
        
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
            
            # Test different layer configurations
            layer_configs = [
                None,  # No hidden states
                [-1],  # Last layer only
                [-3, -2, -1],  # Last 3 layers
                list(range(24, 32))  # Layers 24-31 (typical for larger models)
            ]
            
            prompt_results = {}
            
            # Add concurrency test
            print(f"  Testing concurrency performance...")
            concurrency_results = self.benchmark_concurrency(
                model, prompt, num_concurrent=4, hidden_config={"layers": [-1], "compression": "float16"}
            )
            prompt_results["concurrency"] = concurrency_results
            
            for j, layers in enumerate(layer_configs):
                config_name = f"config_{j}"
                if layers is None:
                    config_name = "no_hidden"
                elif layers == [-1]:
                    config_name = "last_layer"
                elif len(layers) == 3:
                    config_name = "last_3_layers"
                else:
                    config_name = f"layers_{layers[0]}_to_{layers[-1]}"
                
                print(f"  Testing {config_name}...")
                
                if layers is None:
                    result = self.benchmark_generation(model, prompt, None)
                else:
                    compression_results = self.benchmark_compression_modes(
                        model, prompt, layers
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
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        
        # Tokens per second comparison
        ax1.bar(range(len(configs)), tokens_per_sec)
        ax1.set_title('Tokens per Second by Configuration')
        ax1.set_ylabel('Tokens/sec')
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory overhead comparison
        ax2.bar(range(len(configs)), memory_usage)
        ax2.set_title('Memory Overhead by Configuration')
        ax2.set_ylabel('Memory (MB)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Hidden state sizes
        ax3.bar(range(len(configs)), hidden_sizes)
        ax3.set_title('Hidden State Size by Configuration')
        ax3.set_ylabel('Size (MB)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Performance vs Memory scatter
        ax4.scatter(memory_usage, tokens_per_sec, s=60, alpha=0.7)
        ax4.set_xlabel('Memory Overhead (MB)')
        ax4.set_ylabel('Tokens per Second')
        ax4.set_title('Performance vs Memory Trade-off')
        
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
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'benchmark_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

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
        results = benchmark.run_comprehensive_benchmark(
            model="llama3.2:3b",  # Adjust model name as needed
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
