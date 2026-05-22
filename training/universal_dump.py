import torch
from transformers import AutoModel, AutoConfig
import os
import json
import sys

def universal_raw_dump(model_id="gemma-4-E2B-it-Q4_K_M.gguf"):
    """
    Extracts RAW tensors and structural metadata from ANY Hugging Face model.
    This provides the 'Raw Data' for the C++ DeepTranslator to process.
    """
    print(f"--- NCA Universal Raw Dump: {model_id} ---")
    
    # 1. Load configuration first to check dimensions
    print("[1/3] Inspecting Foundation Architecture...")
    try:
        config = AutoConfig.from_pretrained(model_id)
        d_model = getattr(config, "hidden_size", 0)
        num_layers = getattr(config, "num_hidden_layers", 0)
        print(f"  Model: {model_id} | d_model: {d_model} | Layers: {num_layers}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Fetch Raw State Dict
    print("[2/3] Downloading Raw Weight Buffers...")
    try:
        # Using low_cpu_mem_usage and torch_dtype=float32 for maximum fidelity
        model = AutoModel.from_pretrained(
            model_id, 
            torch_dtype=torch.float32, 
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        sd = model.state_dict()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Structural Filtering & Serialization
    # Instead of mapping here, we dump the RAW buffers that the team wants 
    # to 'translate' themselves at the deep level.
    print(f"[3/3] Serializing {len(sd)} weight buffers for C++ DeepTranslator...")
    
    # We save a subset of the first few layers for the 'Silicon-Level' translation work
    raw_dump = {}
    for key, tensor in sd.items():
        # Keep only linear weights and norms (no buffers, no heads)
        if "weight" in key and tensor.dim() >= 1:
            raw_dump[key] = tensor.contiguous()
            print(f"  [+] {key:50} | {tuple(tensor.shape)}")

    output_path = "foundation_raw_dump.pt"
    torch.save(raw_dump, output_path)
    
    # Metadata for the C++ DeepTranslator to understand the source
    metadata = {
        "model_id": model_id,
        "d_model": d_model,
        "num_layers": num_layers,
        "keys": list(raw_dump.keys())
    }
    with open("foundation_meta.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[SUCCESS] Raw Foundation Data ready at {output_path}")
    print("The team can now use DeepTranslator::decompose_to_kronecker or swizzle_to_micro_experts.")

if __name__ == "__main__":
    # Can be gemma, llama, bert, or custom local paths
    target = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-2-2b-it"
    universal_raw_dump(target)
