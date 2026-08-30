# llm_worker.py (Corrected for Positional Embeddings)
import ray
import torch
from transformers import AutoConfig
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class LLM_Layer_Worker:
    def __init__(self, layer_index):
        print(f"[WORKER ACTOR] Initializing for layer {layer_index}...")
        config = AutoConfig.from_pretrained("google/gemma-2b")
        
        # --- THIS IS THE DEFINITIVE FIX ---
        # We are explicitly telling the configuration to use the standard,
        # most compatible attention mechanism. This prevents the KeyError: None.
        config._attn_implementation = "eager"
        # --- END OF FIX ---

        self.layer = GemmaDecoderLayer(config, layer_index)
        print(f"[WORKER ACTOR] Layer {layer_index} structure created.")

    def load_weights(self, weights_dict):
        print(f"[WORKER ACTOR] Receiving and loading weights...")
        self.layer.load_state_dict(weights_dict, strict=True)
        return "Weights loaded successfully."

    def forward(self, hidden_states, position_embeddings):
        print(f"[WORKER ACTOR] Executing forward pass...")
        # We now expect the final, pre-calculated embeddings.
        output = self.layer(hidden_states, position_embeddings=position_embeddings)
        return output