#!/usr/bin/env python3
"""
Safety rails and memory management for hidden states capture.
Prevents OOM conditions and provides intelligent fallback mechanisms.
"""

import psutil
import numpy as np
import json
import time
import threading
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
import gc

class MemoryTier(Enum):
    """Memory usage tiers for adaptive behavior."""
    LOW = "low"           # < 50% memory usage
    MEDIUM = "medium"     # 50-75% memory usage  
    HIGH = "high"         # 75-90% memory usage
    CRITICAL = "critical" # > 90% memory usage

@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_percent: float = 85.0  # Max system memory to use
    max_vram_percent: float = 80.0    # Max VRAM to use
    warning_threshold: float = 75.0   # Warning threshold
    check_interval: float = 1.0       # Memory check interval (seconds)
    auto_fallback: bool = True        # Enable automatic fallbacks
    compression_fallback: bool = True # Enable compression fallbacks
    layer_reduction_fallback: bool = True # Enable layer reduction

@dataclass
class HiddenStateConfig:
    """Configuration for hidden state capture with safety limits."""
    layers: Optional[List[int]] = None
    compression: str = "none"  # none, float16, int8
    max_sequence_length: int = 2048
    max_batch_size: int = 1
    enable_streaming: bool = True
    memory_limit_mb: Optional[float] = None

class MemoryMonitor:
    """Real-time memory monitoring with alerts."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.monitoring = False
        self.callbacks = []
        self.current_tier = MemoryTier.LOW
        self.monitor_thread = None
        
        # Try to import torch for GPU monitoring
        try:
            import torch
            self.torch_available = torch.cuda.is_available()
        except ImportError:
            self.torch_available = False
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current system memory usage (used_percent, available_gb)."""
        memory = psutil.virtual_memory()
        return memory.percent, memory.available / (1024**3)
    
    def get_vram_usage(self) -> Tuple[float, float]:
        """Get current VRAM usage (used_percent, available_gb)."""
        if not self.torch_available:
            return 0.0, 0.0
        
        try:
            import torch
            if torch.cuda.is_available():
                free_mem, total_mem = torch.cuda.mem_get_info()
                used_mem = total_mem - free_mem
                used_percent = (used_mem / total_mem) * 100
                available_gb = free_mem / (1024**3)
                return used_percent, available_gb
        except Exception:
            pass
        
        return 0.0, 0.0
    
    def estimate_hidden_state_memory(self, config: HiddenStateConfig) -> float:
        """Estimate memory requirements for hidden state capture in MB."""
        
        # Base parameters
        hidden_dim = 4096  # Typical transformer hidden dimension
        seq_len = config.max_sequence_length
        batch_size = config.max_batch_size
        
        # Number of layers
        if config.layers is None:
            num_layers = 32  # Assume full model
        else:
            num_layers = len(config.layers)
        
        # Calculate tensor size
        elements_per_layer = batch_size * seq_len * hidden_dim
        total_elements = elements_per_layer * num_layers
        
        # Bytes per element based on compression
        if config.compression == "float16":
            bytes_per_element = 2
        elif config.compression == "int8":
            bytes_per_element = 1
        else:  # float32
            bytes_per_element = 4
        
        total_bytes = total_elements * bytes_per_element
        total_mb = total_bytes / (1024**2)
        
        # Add overhead for processing (30% buffer)
        total_mb *= 1.3
        
        return total_mb
    
    def get_memory_tier(self) -> MemoryTier:
        """Determine current memory tier."""
        mem_percent, _ = self.get_memory_usage()
        vram_percent, _ = self.get_vram_usage()
        
        max_percent = max(mem_percent, vram_percent)
        
        if max_percent >= 90:
            return MemoryTier.CRITICAL
        elif max_percent >= 75:
            return MemoryTier.HIGH
        elif max_percent >= 50:
            return MemoryTier.MEDIUM
        else:
            return MemoryTier.LOW
    
    def register_callback(self, callback: Callable[[MemoryTier], None]):
        """Register callback for memory tier changes."""
        self.callbacks.append(callback)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                new_tier = self.get_memory_tier()
                
                if new_tier != self.current_tier:
                    self.current_tier = new_tier
                    
                    # Notify callbacks
                    for callback in self.callbacks:
                        try:
                            callback(new_tier)
                        except Exception as e:
                            warnings.warn(f"Memory callback error: {e}")
                
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                warnings.warn(f"Memory monitoring error: {e}")
                time.sleep(self.config.check_interval)
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

class SafetyRails:
    """Main safety system for hidden state capture."""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.monitor = MemoryMonitor(self.config)
        self.fallback_history = []
        self.current_restrictions = {}
        
        # Register memory tier callback
        self.monitor.register_callback(self._handle_memory_tier_change)
        
        # Start monitoring
        self.monitor.start_monitoring()
    
    def _handle_memory_tier_change(self, tier: MemoryTier):
        """Handle memory tier changes with automatic fallbacks."""
        
        print(f"Memory tier changed to: {tier.value}")
        
        if tier == MemoryTier.CRITICAL:
            self._apply_critical_fallbacks()
        elif tier == MemoryTier.HIGH:
            self._apply_high_memory_fallbacks()
        elif tier == MemoryTier.MEDIUM:
            self._apply_medium_memory_optimizations()
        else:
            self._clear_restrictions()
    
    def _apply_critical_fallbacks(self):
        """Apply emergency fallbacks for critical memory usage."""
        
        print("🚨 CRITICAL MEMORY - Applying emergency fallbacks")
        
        # Force garbage collection
        gc.collect()
        
        # Disable hidden states entirely
        self.current_restrictions = {
            'disable_hidden_states': True,
            'max_sequence_length': 512,
            'compression': 'int8',
            'layers': None,  # No hidden states
            'reason': 'critical_memory'
        }
        
        self.fallback_history.append({
            'timestamp': time.time(),
            'tier': MemoryTier.CRITICAL,
            'action': 'disable_hidden_states',
            'restrictions': self.current_restrictions.copy()
        })
    
    def _apply_high_memory_fallbacks(self):
        """Apply fallbacks for high memory usage."""
        
        print("⚠️  HIGH MEMORY - Applying fallbacks")
        
        # Aggressive compression and layer reduction
        self.current_restrictions = {
            'disable_hidden_states': False,
            'max_sequence_length': 1024,
            'compression': 'int8',
            'layers': [-1],  # Only last layer
            'max_batch_size': 1,
            'reason': 'high_memory'
        }
        
        self.fallback_history.append({
            'timestamp': time.time(),
            'tier': MemoryTier.HIGH,
            'action': 'aggressive_compression',
            'restrictions': self.current_restrictions.copy()
        })
    
    def _apply_medium_memory_optimizations(self):
        """Apply optimizations for medium memory usage."""
        
        print("📊 MEDIUM MEMORY - Applying optimizations")
        
        # Moderate compression
        self.current_restrictions = {
            'disable_hidden_states': False,
            'max_sequence_length': 1536,
            'compression': 'float16',
            'layers': [-3, -2, -1],  # Last 3 layers
            'max_batch_size': 1,
            'reason': 'medium_memory'
        }
        
        self.fallback_history.append({
            'timestamp': time.time(),
            'tier': MemoryTier.MEDIUM,
            'action': 'moderate_optimization',
            'restrictions': self.current_restrictions.copy()
        })
    
    def _clear_restrictions(self):
        """Clear memory-based restrictions."""
        if self.current_restrictions:
            print("✅ MEMORY OK - Clearing restrictions")
            self.current_restrictions = {}
    
    def validate_config(self, config: HiddenStateConfig) -> Tuple[bool, HiddenStateConfig, str]:
        """Validate and potentially modify hidden state config for safety."""
        
        # Check if hidden states are disabled
        if self.current_restrictions.get('disable_hidden_states', False):
            safe_config = HiddenStateConfig(
                layers=None,
                compression="none",
                max_sequence_length=config.max_sequence_length,
                max_batch_size=config.max_batch_size
            )
            return False, safe_config, "Hidden states disabled due to memory constraints"
        
        # Start with original config
        safe_config = HiddenStateConfig(
            layers=config.layers,
            compression=config.compression,
            max_sequence_length=config.max_sequence_length,
            max_batch_size=config.max_batch_size,
            enable_streaming=config.enable_streaming,
            memory_limit_mb=config.memory_limit_mb
        )
        
        # Apply current restrictions
        if self.current_restrictions:
            if 'compression' in self.current_restrictions:
                safe_config.compression = self.current_restrictions['compression']
            
            if 'layers' in self.current_restrictions:
                safe_config.layers = self.current_restrictions['layers']
            
            if 'max_sequence_length' in self.current_restrictions:
                safe_config.max_sequence_length = min(
                    safe_config.max_sequence_length,
                    self.current_restrictions['max_sequence_length']
                )
            
            if 'max_batch_size' in self.current_restrictions:
                safe_config.max_batch_size = min(
                    safe_config.max_batch_size,
                    self.current_restrictions['max_batch_size']
                )
        
        # Estimate memory requirements
        estimated_mb = self.monitor.estimate_hidden_state_memory(safe_config)
        
        # Check against memory limits
        mem_percent, available_gb = self.monitor.get_memory_usage()
        available_mb = available_gb * 1024
        
        if estimated_mb > available_mb * 0.8:  # Don't use more than 80% of available
            # Try progressive fallbacks
            
            # 1. Try compression
            if safe_config.compression == "none":
                safe_config.compression = "float16"
                estimated_mb = self.monitor.estimate_hidden_state_memory(safe_config)
            
            if estimated_mb > available_mb * 0.8 and safe_config.compression == "float16":
                safe_config.compression = "int8"
                estimated_mb = self.monitor.estimate_hidden_state_memory(safe_config)
            
            # 2. Try reducing layers
            if estimated_mb > available_mb * 0.8 and safe_config.layers:
                if len(safe_config.layers) > 3:
                    safe_config.layers = safe_config.layers[-3:]  # Last 3 layers
                    estimated_mb = self.monitor.estimate_hidden_state_memory(safe_config)
                
                if estimated_mb > available_mb * 0.8:
                    safe_config.layers = [safe_config.layers[-1]]  # Last layer only
                    estimated_mb = self.monitor.estimate_hidden_state_memory(safe_config)
            
            # 3. Try reducing sequence length
            if estimated_mb > available_mb * 0.8:
                safe_config.max_sequence_length = min(safe_config.max_sequence_length, 1024)
                estimated_mb = self.monitor.estimate_hidden_state_memory(safe_config)
            
            if estimated_mb > available_mb * 0.8:
                safe_config.max_sequence_length = min(safe_config.max_sequence_length, 512)
                estimated_mb = self.monitor.estimate_hidden_state_memory(safe_config)
            
            # 4. Last resort - disable hidden states
            if estimated_mb > available_mb * 0.8:
                safe_config.layers = None
                return False, safe_config, f"Memory limit exceeded (estimated {estimated_mb:.1f}MB > available {available_mb * 0.8:.1f}MB)"
        
        # Check if config was modified
        config_modified = (
            safe_config.layers != config.layers or
            safe_config.compression != config.compression or
            safe_config.max_sequence_length != config.max_sequence_length or
            safe_config.max_batch_size != config.max_batch_size
        )
        
        message = ""
        if config_modified:
            message = f"Config modified for safety (estimated memory: {estimated_mb:.1f}MB)"
        
        return True, safe_config, message
    
    def get_recommended_config(self) -> HiddenStateConfig:
        """Get recommended config based on current memory situation."""
        
        tier = self.monitor.get_memory_tier()
        
        if tier == MemoryTier.CRITICAL:
            return HiddenStateConfig(
                layers=None,
                compression="none",
                max_sequence_length=256,
                max_batch_size=1
            )
        elif tier == MemoryTier.HIGH:
            return HiddenStateConfig(
                layers=[-1],
                compression="int8",
                max_sequence_length=512,
                max_batch_size=1
            )
        elif tier == MemoryTier.MEDIUM:
            return HiddenStateConfig(
                layers=[-3, -2, -1],
                compression="float16",
                max_sequence_length=1024,
                max_batch_size=1
            )
        else:  # LOW
            return HiddenStateConfig(
                layers=list(range(-8, 0)),  # Last 8 layers
                compression="none",
                max_sequence_length=2048,
                max_batch_size=1
            )
    
    def get_status_report(self) -> Dict:
        """Get comprehensive status report."""
        
        mem_percent, available_gb = self.monitor.get_memory_usage()
        vram_percent, vram_available_gb = self.monitor.get_vram_usage()
        
        return {
            'memory': {
                'usage_percent': mem_percent,
                'available_gb': available_gb,
                'tier': self.monitor.current_tier.value
            },
            'vram': {
                'usage_percent': vram_percent,
                'available_gb': vram_available_gb,
                'available': self.monitor.torch_available
            },
            'restrictions': self.current_restrictions,
            'fallback_history': self.fallback_history[-5:],  # Last 5 events
            'recommended_config': self.get_recommended_config().__dict__
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.monitor.stop_monitoring()

# Integration with existing hidden states client

class SafeHiddenStatesClient:
    """Hidden states client with integrated safety rails."""
    
    def __init__(self, base_url: str = "http://localhost:11434", safety_config: MemoryConfig = None):
        self.base_url = base_url
        self.safety = SafetyRails(safety_config)
        
    def generate_with_safety(
        self,
        model: str,
        prompt: str,
        hidden_config: HiddenStateConfig,
        **kwargs
    ) -> Tuple[Dict, str]:
        """Generate with automatic safety validation."""
        
        # Validate config
        is_safe, safe_config, message = self.safety.validate_config(hidden_config)
        
        if message:
            print(f"Safety: {message}")
        
        # Prepare request
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            **kwargs
        }
        
        # Add hidden states config if enabled
        if safe_config.layers is not None:
            request_data["hidden_states"] = {
                "layers": safe_config.layers,
                "compression": safe_config.compression
            }
        
        # Make request (simplified - would use actual HTTP client)
        # This is a placeholder for the actual implementation
        response = {
            "text": "Generated text...",
            "hidden_states": {} if safe_config.layers else None,
            "safety_applied": not is_safe,
            "safety_message": message
        }
        
        return response, message
    
    def get_safety_status(self) -> Dict:
        """Get current safety status."""
        return self.safety.get_status_report()
    
    def cleanup(self):
        """Cleanup safety resources."""
        self.safety.cleanup()

# Example usage and testing

def test_safety_rails():
    """Test safety rails functionality."""
    
    print("Testing Safety Rails System...")
    
    # Initialize safety system
    safety = SafetyRails()
    
    # Test different configs
    test_configs = [
        HiddenStateConfig(layers=list(range(32)), compression="none"),  # Large config
        HiddenStateConfig(layers=[-1], compression="float16"),         # Small config
        HiddenStateConfig(layers=None, compression="none"),            # No hidden states
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\nTesting config {i+1}:")
        print(f"  Original: layers={config.layers}, compression={config.compression}")
        
        is_safe, safe_config, message = safety.validate_config(config)
        
        print(f"  Safe: {is_safe}")
        print(f"  Modified: layers={safe_config.layers}, compression={safe_config.compression}")
        print(f"  Message: {message}")
    
    # Print status report
    print("\nStatus Report:")
    status = safety.get_status_report()
    print(json.dumps(status, indent=2, default=str))
    
    # Cleanup
    safety.cleanup()

if __name__ == "__main__":
    test_safety_rails()
