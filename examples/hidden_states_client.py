#!/usr/bin/env python3
"""
Example Python client for Ollama hidden state streaming.

This script demonstrates how to use the hidden state capture functionality
to extract transformer layer activations during text generation.
"""

import json
import requests
import numpy as np
from typing import List, Dict, Any, Optional

class OllamaHiddenStatesClient:
    """Client for interacting with Ollama's hidden state API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def generate_with_hidden_states(
        self,
        model: str,
        prompt: str,
        layers: Optional[List[int]] = None,
        last_layer_only: bool = False,
        compression: str = "none",
        stream: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate text with hidden state capture enabled.
        
        Args:
            model: Name of the model to use
            prompt: Input prompt
            layers: Specific layers to capture (None for all layers)
            last_layer_only: Only capture the final transformer layer
            compression: Compression type ("none", "float16", "int8")
            stream: Whether to stream responses
            
        Returns:
            List of response objects containing text and hidden states
        """
        
        # Configure hidden state capture
        expose_hidden = {
            "enabled": True,
            "layers": layers,
            "last_layer_only": last_layer_only,
            "compression": compression
        }
        
        # Prepare request
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "expose_hidden": expose_hidden
        }
        
        url = f"{self.base_url}/api/generate"
        
        responses = []
        
        try:
            with requests.post(url, json=request_data, stream=stream) as response:
                response.raise_for_status()
                
                if stream:
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            responses.append(data)
                            
                            # Print progress
                            if 'response' in data and data['response']:
                                print(data['response'], end='', flush=True)
                            
                            # Print hidden state info if available
                            if 'hidden_states' in data and data['hidden_states']:
                                print(f"\n[Hidden states captured: {len(data['hidden_states'])} layers]")
                                
                            if data.get('done', False):
                                break
                else:
                    data = response.json()
                    responses.append(data)
                    print(data.get('response', ''))
                    
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return []
        
        return responses
    
    def analyze_hidden_states(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze captured hidden states from generation responses.
        
        Args:
            responses: List of response objects from generate_with_hidden_states
            
        Returns:
            Analysis results including statistics and layer information
        """
        
        all_hidden_states = []
        
        # Collect all hidden states
        for response in responses:
            if 'hidden_states' in response:
                all_hidden_states.extend(response['hidden_states'])
        
        if not all_hidden_states:
            return {"error": "No hidden states found in responses"}
        
        # Analyze by layer
        layer_stats = {}
        
        for state in all_hidden_states:
            layer = state['layer']
            shape = state['shape']
            data = np.array(state['data'], dtype=np.float32)
            
            if layer not in layer_stats:
                layer_stats[layer] = {
                    'count': 0,
                    'shapes': [],
                    'mean_activation': 0,
                    'std_activation': 0,
                    'min_activation': float('inf'),
                    'max_activation': float('-inf')
                }
            
            stats = layer_stats[layer]
            stats['count'] += 1
            stats['shapes'].append(shape)
            stats['mean_activation'] += float(np.mean(data))
            stats['std_activation'] += float(np.std(data))
            stats['min_activation'] = min(stats['min_activation'], float(np.min(data)))
            stats['max_activation'] = max(stats['max_activation'], float(np.max(data)))
        
        # Average the statistics
        for layer, stats in layer_stats.items():
            if stats['count'] > 0:
                stats['mean_activation'] /= stats['count']
                stats['std_activation'] /= stats['count']
        
        return {
            "total_states": len(all_hidden_states),
            "layers_captured": list(layer_stats.keys()),
            "layer_statistics": layer_stats
        }


def main():
    """Example usage of the hidden states client."""
    
    client = OllamaHiddenStatesClient()
    
    print("=== Ollama Hidden States Example ===\n")
    
    # Example 1: Capture last layer only
    print("Example 1: Capturing last layer only")
    print("Prompt: 'The quick brown fox'")
    print("Response: ", end='')
    
    responses = client.generate_with_hidden_states(
        model="llama3.2",  # Adjust model name as needed
        prompt="The quick brown fox",
        last_layer_only=True,
        stream=True
    )
    
    print("\n")
    
    # Analyze the captured states
    analysis = client.analyze_hidden_states(responses)
    print(f"Analysis: {json.dumps(analysis, indent=2)}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Capture specific layers
    print("Example 2: Capturing specific layers [0, 5, 10]")
    print("Prompt: 'Explain quantum computing'")
    print("Response: ", end='')
    
    responses = client.generate_with_hidden_states(
        model="llama3.2",
        prompt="Explain quantum computing",
        layers=[0, 5, 10],
        stream=True
    )
    
    print("\n")
    
    # Analyze the captured states
    analysis = client.analyze_hidden_states(responses)
    print(f"Analysis: {json.dumps(analysis, indent=2)}")


if __name__ == "__main__":
    main()
