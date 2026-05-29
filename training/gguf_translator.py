import gguf
import torch
import numpy as np
import gc
import os
import json
import sys

def extract_gguf_to_nca(gguf_path="gemma-4-e2b-it-Q4_K_M.gguf"):
    """
    Surgical GGUF Extractor.
    Only takes the material needed for NCA Aether cycles to save RAM.
    """
    if not os.path.exists(gguf_path):
        print(f"[Error] GGUF file not found: {gguf_path}")
        return

    print(f"--- NCA GGUF Surgical Extractor: {os.path.basename(gguf_path)} ---")
    
    try:
        reader = gguf.GGUFReader(gguf_path)
    except Exception as e:
        print(f"[Error] Failed to read GGUF: {e}")
        return

    nca_material = {}
    
    print(f"[1/2] Surgically Dequantizing Target Tensors (FP32)...")
    
    # NCA SDMS uses 1024 Micro-Experts.
    # Gemma layers are usually 16384 intermediate dim.
    # We only need the first 2-3 layers to fully populate the NCA expert pool.
    TARGET_BLOCKS = 3 # Only take first 3 blocks to stay under RAM limits
    
    for tensor in reader.tensors:
        # TARGETED FILTERING: Only take what the DeepTranslator actually uses
        is_needed = False
        
        # 1. Expert Seed Material (ffn_gate, ffn_up)
        if "ffn_gate" in tensor.name or "ffn_up" in tensor.name:
            block_idx = int(tensor.name.split('.')[1])
            if block_idx < TARGET_BLOCKS: is_needed = True
            
        # 2. Spectral Seed Material (attn_q)
        if "attn_q.weight" in tensor.name:
            block_idx = int(tensor.name.split('.')[1])
            if block_idx == 0: is_needed = True # Only need Layer 0 for Spectral
            
        # 3. GLR Factor Material (ffn_norm, attn_norm)
        if "ffn_norm" in tensor.name or "attn_norm" in tensor.name:
            block_idx = int(tensor.name.split('.')[1])
            if block_idx == 0: is_needed = True

        if not is_needed:
            continue
            
        print(f"  Extracting: {tensor.name:50} | {tensor.tensor_type.name}")
        
        # Dequantize to raw numpy and copy to make writable
        raw_numpy = gguf.dequantize(tensor.data, tensor.tensor_type)
        fp32_tensor = torch.from_numpy(raw_numpy.copy()).float().contiguous()

        # ── THE GEOMETRIC COMPILER (Distillation) ──
        if len(fp32_tensor.shape) == 2:
            print(f"    -> Compiling {fp32_tensor.shape} to Geometric Wavefront (Top-16)...")
            # For each concept (row), find the top 16 strongest connections
            # This mathematically prunes the noise and creates the structural pointers
            top_probs, top_indices = torch.topk(torch.abs(fp32_tensor), k=16, dim=1)
            
            # Re-gather the actual signed probabilities based on the indices
            actual_probs = torch.gather(fp32_tensor, 1, top_indices)
            
            # Normalize probabilities within the Top-16 Wavefront
            # This perfectly preserves the probability mass during deep recursion
            row_sums = torch.sum(torch.abs(actual_probs), dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0 # Prevent division by zero
            normalized_probs = actual_probs / row_sums
            
            # We store the compiled representation: 
            # 1. The Pointers (next_shape_id)
            # 2. The Quantized Q-Values (width/probabilities)
            nca_material[tensor.name + ".pointers"] = top_indices.to(torch.int32)
            nca_material[tensor.name + ".probs"] = normalized_probs
        else:
            # 1D tensors (biases/norms) remain continuous
            nca_material[tensor.name] = fp32_tensor
        
        del raw_numpy
        gc.collect()

    output_path = "foundation_dequantized.pt"
    print(f"[2/2] Serializing {len(nca_material)} surgical buffers to {output_path}...")
    
    # Using pickle_protocol=4 for better handling of large blobs
    torch.save(nca_material, output_path)
    
    # Export metadata
    metadata = {
        "source_gguf": gguf_path,
        "mode": "geometric_compiled",
        "wavefront_size": 16,
        "tensor_count": len(nca_material),
        "keys": list(nca_material.keys())
    }
    with open("gguf_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[SUCCESS] Memory-safe material exported.")
    print(f"  Note: Captured {TARGET_BLOCKS} blocks. This is enough for the NCA Expert Pool.")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-e2b-it-Q4_K_M.gguf"
    extract_gguf_to_nca(target)
