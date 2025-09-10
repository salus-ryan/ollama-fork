package api

import (
	"encoding/json"
	"testing"
)

func TestHiddenStateConfig(t *testing.T) {
	tests := []struct {
		name   string
		config HiddenStateConfig
		want   string
	}{
		{
			name: "basic config",
			config: HiddenStateConfig{
				Enabled:       true,
				LastLayerOnly: false,
				Compression:   "none",
			},
			want: `{"enabled":true,"last_layer_only":false,"compression":"none"}`,
		},
		{
			name: "with specific layers",
			config: HiddenStateConfig{
				Enabled:     true,
				Layers:      []int{0, 5, 10},
				Compression: "float16",
			},
			want: `{"enabled":true,"layers":[0,5,10],"compression":"float16"}`,
		},
		{
			name: "last layer only",
			config: HiddenStateConfig{
				Enabled:       true,
				LastLayerOnly: true,
				Compression:   "int8",
			},
			want: `{"enabled":true,"last_layer_only":true,"compression":"int8"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.config)
			if err != nil {
				t.Fatalf("Failed to marshal config: %v", err)
			}

			if string(data) != tt.want {
				t.Errorf("Marshal() = %s, want %s", string(data), tt.want)
			}

			// Test unmarshaling
			var config HiddenStateConfig
			if err := json.Unmarshal(data, &config); err != nil {
				t.Fatalf("Failed to unmarshal config: %v", err)
			}

			if config.Enabled != tt.config.Enabled {
				t.Errorf("Enabled = %v, want %v", config.Enabled, tt.config.Enabled)
			}
			if config.LastLayerOnly != tt.config.LastLayerOnly {
				t.Errorf("LastLayerOnly = %v, want %v", config.LastLayerOnly, tt.config.LastLayerOnly)
			}
			if config.Compression != tt.config.Compression {
				t.Errorf("Compression = %v, want %v", config.Compression, tt.config.Compression)
			}
		})
	}
}

func TestHiddenState(t *testing.T) {
	state := HiddenState{
		Layer:       5,
		Shape:       []int{1, 10, 768},
		Data:        []float32{0.1, 0.2, 0.3, 0.4, 0.5},
		TokenIndex:  2,
		Compression: "none",
	}

	data, err := json.Marshal(state)
	if err != nil {
		t.Fatalf("Failed to marshal hidden state: %v", err)
	}

	var unmarshaled HiddenState
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal hidden state: %v", err)
	}

	if unmarshaled.Layer != state.Layer {
		t.Errorf("Layer = %v, want %v", unmarshaled.Layer, state.Layer)
	}
	if len(unmarshaled.Shape) != len(state.Shape) {
		t.Errorf("Shape length = %v, want %v", len(unmarshaled.Shape), len(state.Shape))
	}
	if len(unmarshaled.Data) != len(state.Data) {
		t.Errorf("Data length = %v, want %v", len(unmarshaled.Data), len(state.Data))
	}
	if unmarshaled.TokenIndex != state.TokenIndex {
		t.Errorf("TokenIndex = %v, want %v", unmarshaled.TokenIndex, state.TokenIndex)
	}
	if unmarshaled.Compression != state.Compression {
		t.Errorf("Compression = %v, want %v", unmarshaled.Compression, state.Compression)
	}
}

func TestGenerateRequestWithHiddenStates(t *testing.T) {
	req := GenerateRequest{
		Model:  "test-model",
		Prompt: "Hello world",
		ExposeHidden: &HiddenStateConfig{
			Enabled:       true,
			LastLayerOnly: true,
			Compression:   "float16",
		},
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}

	var unmarshaled GenerateRequest
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal request: %v", err)
	}

	if unmarshaled.ExposeHidden == nil {
		t.Fatal("ExposeHidden is nil after unmarshaling")
	}
	if !unmarshaled.ExposeHidden.Enabled {
		t.Error("ExposeHidden.Enabled should be true")
	}
	if !unmarshaled.ExposeHidden.LastLayerOnly {
		t.Error("ExposeHidden.LastLayerOnly should be true")
	}
	if unmarshaled.ExposeHidden.Compression != "float16" {
		t.Errorf("ExposeHidden.Compression = %v, want float16", unmarshaled.ExposeHidden.Compression)
	}
}

func TestGenerateResponseWithHiddenStates(t *testing.T) {
	resp := GenerateResponse{
		Model:    "test-model",
		Response: "Hello there!",
		Done:     true,
		HiddenStates: []HiddenState{
			{
				Layer:       0,
				Shape:       []int{1, 3, 768},
				Data:        []float32{0.1, 0.2, 0.3},
				TokenIndex:  0,
				Compression: "none",
			},
			{
				Layer:       11,
				Shape:       []int{1, 3, 768},
				Data:        []float32{0.4, 0.5, 0.6},
				TokenIndex:  1,
				Compression: "none",
			},
		},
	}

	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("Failed to marshal response: %v", err)
	}

	var unmarshaled GenerateResponse
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	if len(unmarshaled.HiddenStates) != 2 {
		t.Errorf("HiddenStates length = %v, want 2", len(unmarshaled.HiddenStates))
	}

	// Check first hidden state
	state0 := unmarshaled.HiddenStates[0]
	if state0.Layer != 0 {
		t.Errorf("First state layer = %v, want 0", state0.Layer)
	}
	if len(state0.Data) != 3 {
		t.Errorf("First state data length = %v, want 3", len(state0.Data))
	}

	// Check second hidden state
	state1 := unmarshaled.HiddenStates[1]
	if state1.Layer != 11 {
		t.Errorf("Second state layer = %v, want 11", state1.Layer)
	}
	if state1.TokenIndex != 1 {
		t.Errorf("Second state token index = %v, want 1", state1.TokenIndex)
	}
}
