# Hidden States API

This document describes the hidden states functionality in Ollama, which allows you to capture and stream transformer layer activations during text generation.

## Overview

The hidden states feature enables real-time access to internal transformer representations at each layer during inference. This is useful for:

- Latent space analysis and manipulation
- Model interpretability research
- Real-time braiding of multiple model streams
- Activation visualization and debugging

## API Reference

### Configuration

Hidden state capture is configured using the `ExposeHidden` field in generation requests:

```json
{
  "model": "llama3.2",
  "prompt": "Hello world",
  "expose_hidden": {
    "enabled": true,
    "layers": [0, 5, 10],
    "last_layer_only": false,
    "compression": "float16"
  }
}
```

#### HiddenStateConfig Fields

- `enabled` (bool): Enable/disable hidden state capture
- `layers` ([]int, optional): Specific layers to capture (null for all layers)
- `last_layer_only` (bool): Only capture the final transformer layer
- `compression` (string): Compression type - "none", "float16", or "int8"

### Response Format

Hidden states are included in generation responses:

```json
{
  "model": "llama3.2",
  "response": "Hello",
  "done": false,
  "hidden_states": [
    {
      "layer": 0,
      "shape": [1, 1, 4096],
      "data": [0.1, 0.2, 0.3, ...],
      "token_index": 0,
      "compression": "float16"
    }
  ]
}
```

#### HiddenState Fields

- `layer` (int): Transformer layer index (0-based)
- `shape` ([]int): Tensor dimensions [batch, sequence, hidden_size]
- `data` ([]float32): Flattened activation values
- `token_index` (int): Position in the generated sequence
- `compression` (string): Applied compression type

## Usage Examples

### Basic Usage

```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "prompt": "Explain quantum computing",
    "expose_hidden": {
      "enabled": true,
      "last_layer_only": true
    }
  }'
```

### Python Client

```python
import requests
import json

def generate_with_hidden_states(prompt, model="llama3.2"):
    response = requests.post("http://localhost:11434/api/generate", 
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "expose_hidden": {
                "enabled": True,
                "layers": [0, 5, 10],
                "compression": "float16"
            }
        }, stream=True)
    
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if 'hidden_states' in data:
                print(f"Captured {len(data['hidden_states'])} layer states")
            if data.get('done'):
                break

generate_with_hidden_states("Hello world")
```

### JavaScript Client

```javascript
async function generateWithHiddenStates(prompt) {
    const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: 'llama3.2',
            prompt: prompt,
            stream: true,
            expose_hidden: {
                enabled: true,
                last_layer_only: true,
                compression: 'float16'
            }
        })
    });

    const reader = response.body.getReader();
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const data = JSON.parse(new TextDecoder().decode(value));
        if (data.hidden_states) {
            console.log(`Hidden states: ${data.hidden_states.length} layers`);
        }
    }
}
```

## Performance Considerations

### Memory Usage

Hidden states can consume significant memory:
- Uncompressed: ~4 bytes per activation × hidden_size × num_layers
- Float16: ~2 bytes per activation (50% reduction)
- Int8: ~1 byte per activation (75% reduction)

For a 7B parameter model with 32 layers and 4096 hidden size:
- Uncompressed: ~512 KB per token
- Float16: ~256 KB per token  
- Int8: ~128 KB per token

### Compression Trade-offs

| Compression | Size Reduction | Quality Loss | Use Case |
|-------------|----------------|--------------|----------|
| none        | 0%             | None         | Research, debugging |
| float16     | 50%            | Minimal      | Production, analysis |
| int8        | 75%            | Moderate     | Bandwidth-limited |

### Performance Impact

Enabling hidden state capture adds overhead:
- ~5-10% inference slowdown (uncompressed)
- ~10-15% inference slowdown (with compression)
- Memory allocation overhead for state storage

## Configuration Options

### Layer Selection

```json
// Capture all layers
"expose_hidden": {"enabled": true}

// Capture specific layers
"expose_hidden": {"enabled": true, "layers": [0, 15, 31]}

// Capture only the last layer
"expose_hidden": {"enabled": true, "last_layer_only": true}
```

### Compression Settings

```json
// No compression (full precision)
"expose_hidden": {"enabled": true, "compression": "none"}

// Half precision (recommended)
"expose_hidden": {"enabled": true, "compression": "float16"}

// 8-bit quantization (maximum compression)
"expose_hidden": {"enabled": true, "compression": "int8"}
```

## Error Handling

Common error scenarios:

### Invalid Configuration
```json
{
  "error": "Invalid hidden state configuration: layers must be non-negative"
}
```

### Memory Allocation Failure
```json
{
  "error": "Failed to allocate memory for hidden state capture"
}
```

### Model Not Supported
```json
{
  "error": "Hidden state capture not supported for this model type"
}
```

## Limitations

1. **Model Support**: Currently supports transformer-based models only
2. **Memory Constraints**: Large models may require compression to avoid OOM
3. **Streaming**: Hidden states are only available in streaming mode
4. **Batch Size**: Limited to batch size of 1 for hidden state capture
5. **Quantization**: Int8 compression introduces some precision loss

## Implementation Details

### Architecture

The hidden state capture system consists of:

1. **C Extension** (`llama/hidden_states_ext.cpp`): Low-level tensor extraction
2. **Go Bindings** (`llama/llama.go`): Wrapper functions for C API
3. **Runner Integration** (`runner/llamarunner/runner.go`): State capture orchestration
4. **API Extension** (`api/types.go`): Request/response type definitions
5. **Server Handlers** (`server/routes.go`): HTTP endpoint modifications

### Data Flow

```
llama.cpp → C Extension → Go Bindings → Runner → Server → Client
    ↓           ↓            ↓          ↓        ↓
  Tensors → Extraction → Compression → JSON → Stream
```

### Memory Management

- Hidden states are allocated per-context in C extension
- Automatic cleanup on context destruction
- Go garbage collection handles response objects
- Streaming reduces peak memory usage

## Troubleshooting

### High Memory Usage

1. Enable compression: `"compression": "float16"`
2. Capture fewer layers: `"layers": [31]` (last layer only)
3. Use smaller batch sizes
4. Monitor system memory during generation

### Performance Issues

1. Disable hidden states for production inference
2. Use float16 compression for balance of speed/quality
3. Capture only necessary layers
4. Consider model size vs. available resources

### Integration Issues

1. Ensure model supports transformer architecture
2. Check API version compatibility
3. Verify streaming is enabled in client
4. Monitor server logs for allocation errors

## Future Enhancements

Planned improvements:

- [ ] Gradient capture support
- [ ] Attention weight extraction
- [ ] Multi-batch hidden state capture
- [ ] Advanced compression algorithms
- [ ] Real-time state manipulation APIs
- [ ] Integration with visualization tools
