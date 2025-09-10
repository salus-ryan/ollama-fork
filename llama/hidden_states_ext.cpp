#include "hidden_states_ext.h"
#include "llama.h"
#include "ggml.h"
#include <vector>
#include <map>
#include <memory>
#include <cstring>
#include <cstdlib>

// Global storage for hidden state configurations and results
static std::map<struct llama_context*, llama_hidden_config> g_hidden_configs;
static std::map<struct llama_context*, std::vector<llama_hidden_state>> g_hidden_states;

// Hook function to capture hidden states during forward pass
// This would need to be integrated into llama.cpp's forward pass
static void capture_hidden_state(struct llama_context* ctx, int layer, int seq_id, 
                                 const struct ggml_tensor* hidden_tensor) {
    auto config_it = g_hidden_configs.find(ctx);
    if (config_it == g_hidden_configs.end() || !config_it->second.enabled) {
        return;
    }
    
    const auto& config = config_it->second;
    
    // Check if we should capture this layer
    bool should_capture = false;
    if (config.last_layer_only) {
        // Get model info to check if this is the last layer
        const struct llama_model* model = llama_get_model(ctx);
        int n_layers = llama_model_n_layers(model);
        should_capture = (layer == n_layers - 1);
    } else if (config.layers == nullptr) {
        // Capture all layers
        should_capture = true;
    } else {
        // Check if layer is in the specified list
        for (int i = 0; i < config.num_layers; i++) {
            if (config.layers[i] == layer) {
                should_capture = true;
                break;
            }
        }
    }
    
    if (!should_capture) {
        return;
    }
    
    // Extract tensor data
    llama_hidden_state state = {};
    state.layer = layer;
    state.token_index = seq_id; // Simplified - would need proper token indexing
    
    // Get tensor shape
    int ndims = ggml_n_dims(hidden_tensor);
    state.shape_len = ndims;
    
    // Calculate total elements
    int total_elements = 1;
    for (int i = 0; i < ndims; i++) {
        total_elements *= state.shape[i];
    }
    
    // Copy tensor data
    state.data_len = total_elements;
    state.data = (float*)malloc(total_elements * sizeof(float));
    
    // Convert from tensor format to float32
    if (hidden_tensor->type == GGML_TYPE_F32) {
        memcpy(state.data, hidden_tensor->data, total_elements * sizeof(float));
    } else if (hidden_tensor->type == GGML_TYPE_F16) {
        // Convert from float16 to float32
        const ggml_fp16_t* src = (const ggml_fp16_t*)hidden_tensor->data;
        for (int i = 0; i < total_elements; i++) {
            state.data[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else {
        // Handle other tensor types as needed
        free(state.data);
        free(state.shape);
        return;
    }
    
    // Apply compression if requested
    if (config.compression == 1) {
        // Float16 compression
        size_t compressed_size = total_elements * sizeof(uint16_t);
        uint16_t* compressed_data = (uint16_t*)malloc(compressed_size);
        if (compressed_data) {
            for (int i = 0; i < total_elements; i++) {
                // Simple float32 to float16 conversion
                float val = state.data[i];
                uint32_t bits = *(uint32_t*)&val;
                uint32_t sign = (bits >> 31) & 0x1;
                uint32_t exp = (bits >> 23) & 0xFF;
                uint32_t frac = bits & 0x7FFFFF;
                
                if (exp == 0) {
                    compressed_data[i] = (uint16_t)(sign << 15);
                } else if (exp == 0xFF) {
                    compressed_data[i] = (uint16_t)((sign << 15) | 0x7C00 | (frac ? 0x200 : 0));
                } else {
                    int new_exp = exp - 127 + 15;
                    if (new_exp <= 0) {
                        compressed_data[i] = (uint16_t)(sign << 15);
                    } else if (new_exp >= 31) {
                        compressed_data[i] = (uint16_t)((sign << 15) | 0x7C00);
                    } else {
                        compressed_data[i] = (uint16_t)((sign << 15) | (new_exp << 10) | (frac >> 13));
                    }
                }
            }
            free(state.data);
            state.data = (float*)compressed_data;
            state.data_len = compressed_size / sizeof(float);
        }
    } else if (config.compression == 2) {
        // Int8 quantization
        float min_val = state.data[0], max_val = state.data[0];
        for (int i = 1; i < total_elements; i++) {
            if (state.data[i] < min_val) min_val = state.data[i];
            if (state.data[i] > max_val) max_val = state.data[i];
        }
        
        float scale = (max_val - min_val) / 255.0f;
        int8_t* quantized_data = (int8_t*)malloc(total_elements * sizeof(int8_t));
        if (quantized_data) {
            for (int i = 0; i < total_elements; i++) {
                float normalized = (state.data[i] - min_val) / scale;
                quantized_data[i] = (int8_t)(normalized - 128);
            }
            free(state.data);
            state.data = (float*)quantized_data;
            state.data_len = total_elements / 4; // Approximate size reduction
        }
    }
    
    // Store the hidden state
    g_hidden_states[ctx].push_back(state);
}

void llama_set_hidden_config(struct llama_context* ctx, const llama_hidden_config* config) {
    if (config == nullptr) {
        g_hidden_configs.erase(ctx);
        return;
    }
    
    llama_hidden_config cfg = *config;
    
    // Deep copy the layers array if provided
    if (config->layers != nullptr && config->num_layers > 0) {
        cfg.layers = (int*)malloc(config->num_layers * sizeof(int));
        memcpy(cfg.layers, config->layers, config->num_layers * sizeof(int));
    }
    
    g_hidden_configs[ctx] = cfg;
    
    // Clear any existing states
    auto states_it = g_hidden_states.find(ctx);
    if (states_it != g_hidden_states.end()) {
        for (auto& state : states_it->second) {
            free(state.data);
            free(state.shape);
        }
        states_it->second.clear();
    }
}

llama_hidden_result* llama_get_hidden_states(struct llama_context* ctx, int seq_id) {
    auto states_it = g_hidden_states.find(ctx);
    if (states_it == g_hidden_states.end() || states_it->second.empty()) {
        return nullptr;
    }
    
    llama_hidden_result* result = (llama_hidden_result*)malloc(sizeof(llama_hidden_result));
    result->num_states = states_it->second.size();
    result->states = (llama_hidden_state*)malloc(result->num_states * sizeof(llama_hidden_state));
    
    // Copy states (shallow copy - caller takes ownership of data)
    for (int i = 0; i < result->num_states; i++) {
        result->states[i] = states_it->second[i];
    }
    
    // Clear the stored states (ownership transferred)
    states_it->second.clear();
    
    return result;
}

void llama_free_hidden_result(llama_hidden_result* result) {
    if (result == nullptr) {
        return;
    }
    
    for (int i = 0; i < result->num_states; i++) {
        free(result->states[i].data);
        free(result->states[i].shape);
    }
    
    free(result->states);
    free(result);
}

bool llama_is_hidden_enabled(struct llama_context* ctx) {
    auto config_it = g_hidden_configs.find(ctx);
    return config_it != g_hidden_configs.end() && config_it->second.enabled;
}

// Cleanup function to be called when context is destroyed
void llama_cleanup_hidden_states(struct llama_context* ctx) {
    // Clean up configuration
    auto config_it = g_hidden_configs.find(ctx);
    if (config_it != g_hidden_configs.end()) {
        if (config_it->second.layers != nullptr) {
            free(config_it->second.layers);
        }
        g_hidden_configs.erase(config_it);
    }
    
    // Clean up stored states
    auto states_it = g_hidden_states.find(ctx);
    if (states_it != g_hidden_states.end()) {
        for (auto& state : states_it->second) {
            free(state.data);
            free(state.shape);
        }
        g_hidden_states.erase(states_it);
    }
}
