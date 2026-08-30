# llm_master.py (Definitive, Final Version with eager attention)
import json
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import ray
import traceback

from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
from transformers import AutoConfig

from new_worker import LLM_Layer_Worker

# --- 1. INITIALIZE RAY ---
print("[MASTER] Initializing Ray...")
ray.init()
print("[MASTER] Ray is running.")

# --- 2. MAIN EXECUTION BLOCK ---
try:
    # --- Part A: Loader (Unchanged) ---
    LAYER_TO_LOAD = "model.layers.10"
    print(f"[MASTER] Preparing to load weights for: {LAYER_TO_LOAD}")
    
    model_index_path = hf_hub_download(repo_id="google/gemma-2b", filename="model.safetensors.index.json")
    with open(model_index_path, "r") as f: model_index = json.load(f)
    
    weights_file = model_index["weight_map"].get(f"{LAYER_TO_LOAD}.self_attn.q_proj.weight")
    if not weights_file: raise ValueError(f"Could not find weights file for {LAYER_TO_LOAD}")
    
    cached_path = hf_hub_download(repo_id="google/gemma-2b", filename=weights_file)
    layer_weights = {}
    with safe_open(cached_path, framework="pt", device="cpu") as f:
        prefix_len = len(LAYER_TO_LOAD) + 1
        for key in f.keys():
            if key.startswith(LAYER_TO_LOAD):
                layer_weights[key[prefix_len:]] = f.get_tensor(key)
    if not layer_weights: raise ValueError(f"No tensors found for {LAYER_TO_LOAD}")
    print(f"[MASTER] Successfully extracted {len(layer_weights)} tensors.")

    # --- Part B: Ray Logic (Corrected) ---
    layer_index = int(LAYER_TO_LOAD.split('.')[2])
    print(f"[MASTER] Creating remote worker Actor for layer {layer_index}...")
    remote_worker = LLM_Layer_Worker.remote(layer_index=layer_index)
    status = ray.get(remote_worker.load_weights.remote(layer_weights))
    print(f"[MASTER] Status from worker: '{status}'")

    print("[MASTER] Preparing dummy input AND positional embeddings...")
    config = AutoConfig.from_pretrained("google/gemma-2b")
    config._attn_implementation = "eager"
    
    batch_size = 1; sequence_length = 8
    dummy_input = torch.randn(batch_size, sequence_length, config.hidden_size)

    # This is the correct calculation logic
    rotary_emb = GemmaRotaryEmbedding(config)
    position_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary_emb(dummy_input, position_ids)

    print("[MASTER] Sending input AND final embeddings to worker for forward pass...")
    output_ref = remote_worker.forward.remote(dummy_input, position_embeddings)
    output_tensor = ray.get(output_ref)
    print("\n--- INFERENCE COMPLETE ---")
    print("Input shape:", dummy_input.shape)
    print("Output shape from remote worker:", output_tensor.shape)
    if dummy_input.shape == output_tensor.shape:
        print("✅ Success! The shapes match. The remote layer is working correctly.")
    else:
        print("❌ Error! The output shape does not match the input shape.")

except Exception as e:
    traceback.print_exc()
    print(f"[MASTER] FATAL ERROR during execution: {e}")

finally:
    print("[MASTER] Shutting down Ray cluster connection.")
    ray.shutdown()