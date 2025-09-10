#ifndef HIDDEN_STATES_EXT_H
#define HIDDEN_STATES_EXT_H

#include "llama.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Configuration for hidden state capture
typedef struct {
    bool enabled;
    int* layers;           // Array of layer indices to capture (NULL for all)
    int num_layers;        // Number of layers in the array
    bool last_layer_only;  // Capture only the last layer
    int compression;       // 0=none, 1=float16, 2=int8
} llama_hidden_config;

// Hidden state data structure
typedef struct {
    int layer;             // Layer index
    int token_index;       // Token position in sequence
    int* shape;            // Tensor shape [batch_size, seq_len, hidden_size]
    int shape_len;         // Number of dimensions
    float* data;           // Flattened tensor data
    int data_len;          // Number of elements in data
    int compression;       // Compression type used
} llama_hidden_state;

// Hidden state capture result
typedef struct {
    llama_hidden_state* states;  // Array of hidden states
    int num_states;              // Number of captured states
} llama_hidden_result;

// Enable hidden state capture for a context
void llama_set_hidden_config(struct llama_context* ctx, const llama_hidden_config* config);

// Get captured hidden states after decode
llama_hidden_result* llama_get_hidden_states(struct llama_context* ctx, int seq_id);

// Free hidden state result
void llama_free_hidden_result(llama_hidden_result* result);

// Check if hidden state capture is enabled
bool llama_is_hidden_enabled(struct llama_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // HIDDEN_STATES_EXT_H
