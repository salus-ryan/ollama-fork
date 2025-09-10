#!/usr/bin/env python3
"""
Braiding layer prototype for hidden state fusion.
Implements logit fusion and whitened + projected hidden state fusion with orthogonal-covariance math.
"""

import numpy as np
import json
import requests
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import asyncio
import aiohttp
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

@dataclass
class BraidingResult:
    """Results from a braiding operation."""
    fused_logits: Optional[np.ndarray] = None
    fused_hidden_states: Optional[Dict[str, np.ndarray]] = None
    fusion_weights: Optional[np.ndarray] = None
    quality_metrics: Optional[Dict[str, float]] = None
    orthogonal_transform: Optional[np.ndarray] = None

class LogitFusion:
    """Clean logit fusion for vocab-aligned braiding."""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.fusion_history = []
    
    def weighted_average_fusion(
        self, 
        logits_list: List[np.ndarray], 
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Simple weighted averaging of logits (baseline approach)."""
        
        if not logits_list:
            raise ValueError("Empty logits list")
        
        # Normalize weights
        if weights is None:
            weights = [1.0 / len(logits_list)] * len(logits_list)
        else:
            weights = np.array(weights)
            weights = weights / np.sum(weights)
        
        # Ensure all logits have same shape
        target_shape = logits_list[0].shape
        for i, logits in enumerate(logits_list):
            if logits.shape != target_shape:
                raise ValueError(f"Logits shape mismatch: {logits.shape} vs {target_shape}")
        
        # Weighted sum
        fused_logits = np.zeros_like(logits_list[0])
        for logits, weight in zip(logits_list, weights):
            fused_logits += weight * logits
        
        return fused_logits
    
    def entropy_weighted_fusion(
        self, 
        logits_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Entropy-weighted fusion - higher confidence gets more weight."""
        
        if not logits_list:
            raise ValueError("Empty logits list")
        
        # Calculate entropy for each logit distribution
        entropies = []
        for logits in logits_list:
            # Convert to probabilities
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            # Calculate entropy (lower entropy = higher confidence)
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
            entropies.append(np.mean(entropy))
        
        # Convert to weights (inverse entropy)
        entropies = np.array(entropies)
        weights = 1.0 / (entropies + 1e-10)
        weights = weights / np.sum(weights)
        
        fused_logits = self.weighted_average_fusion(logits_list, weights.tolist())
        
        return fused_logits, weights
    
    def top_k_consensus_fusion(
        self, 
        logits_list: List[np.ndarray], 
        k: int = 10
    ) -> np.ndarray:
        """Fusion based on top-k consensus across models."""
        
        if not logits_list:
            raise ValueError("Empty logits list")
        
        # Get top-k indices for each model
        top_k_indices = []
        for logits in logits_list:
            if len(logits.shape) == 1:
                top_k = np.argsort(logits)[-k:]
            else:
                # Handle batch dimension
                top_k = np.argsort(logits, axis=-1)[..., -k:]
            top_k_indices.append(top_k)
        
        # Find consensus tokens (appear in multiple top-k lists)
        if len(logits_list[0].shape) == 1:
            # Single sequence case
            consensus_tokens = set(top_k_indices[0])
            for top_k in top_k_indices[1:]:
                consensus_tokens = consensus_tokens.intersection(set(top_k))
            
            # Weight consensus tokens more heavily
            fused_logits = np.mean(logits_list, axis=0)
            consensus_boost = 1.5
            for token_id in consensus_tokens:
                fused_logits[token_id] *= consensus_boost
        else:
            # Batch case - simplified to weighted average
            fused_logits = np.mean(logits_list, axis=0)
        
        return fused_logits

class HiddenStateFusion:
    """Advanced hidden state fusion with orthogonalization and projection."""
    
    def __init__(self, hidden_dim: int = 4096):
        self.hidden_dim = hidden_dim
        self.whitening_transforms = {}
        self.projection_matrices = {}
        self.alignment_history = []
    
    def compute_covariance_matrix(self, hidden_states: np.ndarray) -> np.ndarray:
        """Compute covariance matrix for whitening."""
        # hidden_states shape: (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        
        if len(hidden_states.shape) == 3:
            # Flatten batch and sequence dimensions
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        
        # Center the data
        mean = np.mean(hidden_states, axis=0)
        centered = hidden_states - mean
        
        # Compute covariance
        cov_matrix = np.cov(centered.T)
        
        return cov_matrix, mean
    
    def whiten_hidden_states(
        self, 
        hidden_states: np.ndarray, 
        layer_name: str,
        regularization: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply whitening transformation to hidden states."""
        
        original_shape = hidden_states.shape
        
        # Compute or retrieve whitening transform
        if layer_name not in self.whitening_transforms:
            cov_matrix, mean = self.compute_covariance_matrix(hidden_states)
            
            # Eigendecomposition for whitening
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # Add regularization to prevent numerical issues
            eigenvals = np.maximum(eigenvals, regularization)
            
            # Whitening matrix: W = V * D^(-1/2) * V^T
            whitening_matrix = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
            
            self.whitening_transforms[layer_name] = {
                'matrix': whitening_matrix,
                'mean': mean,
                'eigenvals': eigenvals,
                'eigenvecs': eigenvecs
            }
        
        transform = self.whitening_transforms[layer_name]
        
        # Apply whitening
        if len(original_shape) == 3:
            # Batch case
            flat_states = hidden_states.reshape(-1, original_shape[-1])
            centered = flat_states - transform['mean']
            whitened = centered @ transform['matrix']
            whitened = whitened.reshape(original_shape)
        else:
            # Single sequence case
            centered = hidden_states - transform['mean']
            whitened = centered @ transform['matrix']
        
        return whitened, transform['matrix']
    
    def orthogonal_alignment(
        self, 
        hidden_states_a: np.ndarray, 
        hidden_states_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find optimal orthogonal transformation to align hidden states."""
        
        # Flatten if needed
        original_shape_a = hidden_states_a.shape
        original_shape_b = hidden_states_b.shape
        
        if len(original_shape_a) == 3:
            flat_a = hidden_states_a.reshape(-1, original_shape_a[-1])
            flat_b = hidden_states_b.reshape(-1, original_shape_b[-1])
        else:
            flat_a = hidden_states_a
            flat_b = hidden_states_b
        
        # Use Procrustes analysis to find optimal orthogonal transformation
        try:
            R, scale = orthogonal_procrustes(flat_a, flat_b)
            
            # Apply transformation
            aligned_a = flat_a @ R * scale
            
            # Reshape back
            if len(original_shape_a) == 3:
                aligned_a = aligned_a.reshape(original_shape_a)
            
            return aligned_a, R
            
        except Exception as e:
            warnings.warn(f"Orthogonal alignment failed: {e}, using identity")
            return hidden_states_a, np.eye(hidden_states_a.shape[-1])
    
    def project_to_common_space(
        self, 
        hidden_states_list: List[np.ndarray],
        target_dim: Optional[int] = None
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Project all hidden states to a common lower-dimensional space."""
        
        # Determine available dimensions from the data
        sample_states = hidden_states_list[0]
        if len(sample_states.shape) == 3:
            available_features = sample_states.shape[-1]
            total_samples = sum(states.reshape(-1, states.shape[-1]).shape[0] for states in hidden_states_list)
        else:
            available_features = sample_states.shape[-1]
            total_samples = sum(states.shape[0] for states in hidden_states_list)
        
        if target_dim is None:
            target_dim = min(256, available_features // 4, total_samples - 1)  # Safe default
        
        # Stack all hidden states for PCA
        all_states = []
        shapes = []
        
        for states in hidden_states_list:
            original_shape = states.shape
            shapes.append(original_shape)
            
            if len(original_shape) == 3:
                flat_states = states.reshape(-1, original_shape[-1])
            else:
                flat_states = states
            
            all_states.append(flat_states)
        
        # Concatenate all states
        combined_states = np.vstack(all_states)
        
        # Fit PCA
        pca = PCA(n_components=target_dim)
        pca.fit(combined_states)
        
        # Project each hidden state set
        projected_states = []
        start_idx = 0
        
        for i, (states, original_shape) in enumerate(zip(all_states, shapes)):
            end_idx = start_idx + states.shape[0]
            
            # Project to common space
            projected = pca.transform(states)
            
            # Reshape back if needed
            if len(original_shape) == 3:
                projected = projected.reshape(original_shape[0], original_shape[1], target_dim)
            
            projected_states.append(projected)
            start_idx = end_idx
        
        return projected_states, pca.components_
    
    def orthogonal_covariance_fusion(
        self, 
        hidden_states_list: List[np.ndarray],
        layer_names: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """Advanced fusion using orthogonal-covariance approach."""
        
        if len(hidden_states_list) != len(layer_names):
            raise ValueError("Number of hidden states must match number of layer names")
        
        fusion_info = {
            'whitening_applied': [],
            'alignment_transforms': [],
            'projection_matrix': None,
            'fusion_weights': []
        }
        
        # Step 1: Whiten each hidden state set
        whitened_states = []
        for states, layer_name in zip(hidden_states_list, layer_names):
            whitened, transform = self.whiten_hidden_states(states, layer_name)
            whitened_states.append(whitened)
            fusion_info['whitening_applied'].append(layer_name)
        
        # Step 2: Orthogonal alignment (align all to first one)
        aligned_states = [whitened_states[0]]  # First one is reference
        
        for i in range(1, len(whitened_states)):
            aligned, R = self.orthogonal_alignment(whitened_states[i], whitened_states[0])
            aligned_states.append(aligned)
            fusion_info['alignment_transforms'].append(R)
        
        # Step 3: Project to common space
        projected_states, projection_matrix = self.project_to_common_space(aligned_states)
        fusion_info['projection_matrix'] = projection_matrix
        
        # Step 4: Compute fusion weights based on explained variance
        weights = []
        for states in projected_states:
            # Use variance as a proxy for information content
            variance = np.var(states)
            weights.append(variance)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        fusion_info['fusion_weights'] = weights.tolist()
        
        # Step 5: Weighted fusion
        fused_states = np.zeros_like(projected_states[0])
        for states, weight in zip(projected_states, weights):
            fused_states += weight * states
        
        return fused_states, fusion_info

class BraidingLayer:
    """Main braiding layer that combines logit and hidden state fusion."""
    
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 4096):
        self.logit_fusion = LogitFusion(vocab_size)
        self.hidden_fusion = HiddenStateFusion(hidden_dim)
        self.braiding_history = []
    
    def braid_responses(
        self,
        responses: List[Dict],
        fusion_strategy: str = "entropy_weighted",
        hidden_fusion_strategy: str = "orthogonal_covariance"
    ) -> BraidingResult:
        """Main braiding function that fuses multiple model responses."""
        
        # Extract logits and hidden states
        logits_list = []
        hidden_states_dict = {}
        
        for i, response in enumerate(responses):
            if 'logits' in response:
                logits_list.append(np.array(response['logits']))
            
            if 'hidden_states' in response:
                for layer_name, layer_data in response['hidden_states'].items():
                    if layer_name not in hidden_states_dict:
                        hidden_states_dict[layer_name] = []
                    hidden_states_dict[layer_name].append(np.array(layer_data))
        
        result = BraidingResult()
        
        # Fuse logits
        if logits_list:
            if fusion_strategy == "weighted_average":
                result.fused_logits = self.logit_fusion.weighted_average_fusion(logits_list)
            elif fusion_strategy == "entropy_weighted":
                result.fused_logits, result.fusion_weights = self.logit_fusion.entropy_weighted_fusion(logits_list)
            elif fusion_strategy == "top_k_consensus":
                result.fused_logits = self.logit_fusion.top_k_consensus_fusion(logits_list)
            else:
                raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Fuse hidden states
        if hidden_states_dict:
            result.fused_hidden_states = {}
            
            for layer_name, states_list in hidden_states_dict.items():
                if len(states_list) > 1:
                    if hidden_fusion_strategy == "orthogonal_covariance":
                        layer_names = [f"{layer_name}_{i}" for i in range(len(states_list))]
                        fused_layer, fusion_info = self.hidden_fusion.orthogonal_covariance_fusion(
                            states_list, layer_names
                        )
                        result.fused_hidden_states[layer_name] = fused_layer
                        
                        if result.quality_metrics is None:
                            result.quality_metrics = {}
                        result.quality_metrics[f"{layer_name}_fusion_info"] = fusion_info
                    else:
                        # Simple average fallback
                        result.fused_hidden_states[layer_name] = np.mean(states_list, axis=0)
                else:
                    result.fused_hidden_states[layer_name] = states_list[0]
        
        # Store braiding history
        self.braiding_history.append({
            'timestamp': time.time(),
            'num_responses': len(responses),
            'fusion_strategy': fusion_strategy,
            'hidden_fusion_strategy': hidden_fusion_strategy,
            'result_summary': {
                'has_fused_logits': result.fused_logits is not None,
                'num_fused_layers': len(result.fused_hidden_states) if result.fused_hidden_states else 0
            }
        })
        
        return result
    
    def evaluate_braiding_quality(
        self,
        original_responses: List[Dict],
        braided_result: BraidingResult,
        reference_text: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate the quality of braiding results."""
        
        metrics = {}
        
        # Logit diversity (how much the original logits differed)
        if len(original_responses) > 1 and all('logits' in r for r in original_responses):
            logits_arrays = [np.array(r['logits']) for r in original_responses]
            
            # Calculate pairwise KL divergences
            kl_divergences = []
            for i in range(len(logits_arrays)):
                for j in range(i + 1, len(logits_arrays)):
                    # Convert to probabilities
                    p = np.exp(logits_arrays[i] - np.max(logits_arrays[i]))
                    p = p / np.sum(p)
                    q = np.exp(logits_arrays[j] - np.max(logits_arrays[j]))
                    q = q / np.sum(q)
                    
                    # KL divergence
                    kl = np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
                    kl_divergences.append(kl)
            
            metrics['avg_logit_kl_divergence'] = np.mean(kl_divergences)
            metrics['max_logit_kl_divergence'] = np.max(kl_divergences)
        
        # Hidden state alignment quality
        if braided_result.quality_metrics:
            for key, value in braided_result.quality_metrics.items():
                if isinstance(value, dict) and 'fusion_weights' in value:
                    weights = np.array(value['fusion_weights'])
                    # Entropy of fusion weights (lower = more concentrated, potentially better)
                    weight_entropy = -np.sum(weights * np.log(weights + 1e-10))
                    metrics[f'{key}_weight_entropy'] = weight_entropy
        
        # Coherence metrics (if reference text provided)
        if reference_text and braided_result.fused_logits is not None:
            # This would require implementing text generation from logits
            # For now, we'll use a placeholder
            metrics['coherence_score'] = 0.5  # Placeholder
        
        return metrics

# Utility functions for integration with Ollama

async def capture_streaming_triples(
    model: str,
    prompt: str,
    base_url: str = "http://localhost:11434",
    hidden_config: Optional[Dict] = None
) -> List[Dict]:
    """Capture (token, logits, hidden_state) triples from streaming response."""
    
    request_data = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.7}
    }
    
    if hidden_config:
        request_data["hidden_states"] = hidden_config
    
    triples = []
    
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{base_url}/api/generate", json=request_data) as response:
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line)
                        
                        triple = {
                            "token": data.get("response", ""),
                            "logits": data.get("logits"),
                            "hidden_states": data.get("hidden_states")
                        }
                        
                        if any(triple.values()):  # Only add if has some data
                            triples.append(triple)
                        
                        if data.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
    
    return triples

def save_triples_to_jsonl(triples: List[Dict], output_file: str):
    """Save captured triples to JSONL format for later experiments."""
    
    with open(output_file, 'w') as f:
        for triple in triples:
            json.dump(triple, f)
            f.write('\n')

# Example usage and testing

async def test_braiding_layer():
    """Test the braiding layer with sample data."""
    
    # Create sample responses (would come from actual model calls)
    sample_responses = [
        {
            "logits": np.random.randn(32000).tolist(),
            "hidden_states": {
                "layer_24": np.random.randn(128, 4096).tolist(),
                "layer_32": np.random.randn(128, 4096).tolist()
            }
        },
        {
            "logits": np.random.randn(32000).tolist(),
            "hidden_states": {
                "layer_24": np.random.randn(128, 4096).tolist(),
                "layer_32": np.random.randn(128, 4096).tolist()
            }
        }
    ]
    
    # Initialize braiding layer
    braider = BraidingLayer()
    
    # Test different fusion strategies
    strategies = ["weighted_average", "entropy_weighted", "top_k_consensus"]
    
    for strategy in strategies:
        print(f"\nTesting {strategy} fusion...")
        
        result = braider.braid_responses(
            sample_responses,
            fusion_strategy=strategy,
            hidden_fusion_strategy="orthogonal_covariance"
        )
        
        print(f"  Fused logits shape: {result.fused_logits.shape if result.fused_logits is not None else None}")
        print(f"  Fused hidden layers: {list(result.fused_hidden_states.keys()) if result.fused_hidden_states else None}")
        
        # Evaluate quality
        quality = braider.evaluate_braiding_quality(sample_responses, result)
        print(f"  Quality metrics: {quality}")

if __name__ == "__main__":
    import time
    
    print("Testing Braiding Layer...")
    asyncio.run(test_braiding_layer())
