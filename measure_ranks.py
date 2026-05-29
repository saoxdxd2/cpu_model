"""
NCA — Empirical Rank Measurement
Measures the actual SVD rank needed per layer to preserve 99.99% energy.
This tells us exactly how much compression the curve representation gives us.
"""
import gguf
import torch
import numpy as np
import gc
import sys

def measure_ranks(gguf_path):
    print("=" * 64)
    print(" NCA CURVE ANALYSIS: Empirical Rank Measurement")
    print(" Measuring SVD energy spectrum of trained weights")
    print("=" * 64)
    
    reader = gguf.GGUFReader(gguf_path)
    
    # Only measure 2D weight matrices from first 3 blocks (RAM limit)
    ENERGY_THRESHOLD = 0.9999  # 99.99% energy preservation
    
    results = []
    
    for tensor in reader.tensors:
        # Only look at 2D weights (skip norms, biases)
        if not tensor.name.startswith("blk."):
            continue
        block_idx = int(tensor.name.split('.')[1])
        if block_idx > 2:  # First 3 blocks only
            continue
            
        raw = gguf.dequantize(tensor.data, tensor.tensor_type)
        W = torch.from_numpy(raw.copy()).float()
        
        if len(W.shape) != 2:
            continue
        
        m, n = W.shape
        if min(m, n) < 16:  # Skip tiny tensors
            continue
            
        print(f"\n  {tensor.name}")
        print(f"    Shape: [{m} x {n}]  |  Dense size: {m*n*4/1024/1024:.1f} MB")
        
        # Compute SVD
        try:
            S = torch.linalg.svdvals(W)
        except Exception as e:
            print(f"    SVD failed: {e}")
            continue
        
        # Energy spectrum
        energy = (S ** 2)
        total_energy = energy.sum().item()
        cumulative = energy.cumsum(0) / total_energy
        
        # Find rank at different thresholds
        for threshold in [0.99, 0.999, 0.9999, 0.99999]:
            rank = int((cumulative < threshold).sum().item()) + 1
            rank = min(rank, min(m, n))
            curve_size = (m * rank + rank + rank * n) * 4  # U + S + Vt in bytes
            dense_size = m * n * 4
            ratio = dense_size / curve_size
            
            if threshold == ENERGY_THRESHOLD:
                results.append({
                    'name': tensor.name,
                    'shape': (m, n),
                    'rank': rank,
                    'full_rank': min(m, n),
                    'ratio': ratio,
                    'dense_mb': dense_size / 1024 / 1024,
                    'curve_mb': curve_size / 1024 / 1024,
                })
            
            pct = threshold * 100
            print(f"    {pct:8.3f}% energy -> rank {rank:4d} / {min(m,n):4d}  "
                  f"({rank/min(m,n)*100:5.1f}%)  "
                  f"curve: {curve_size/1024/1024:.2f} MB  "
                  f"compression: {ratio:.1f}x")
        
        # Show top singular values
        top5 = S[:5].tolist()
        print(f"    Top-5 singular values: {[f'{v:.2f}' for v in top5]}")
        print(f"    S[0]/S[-1] condition: {S[0]/S[-1]:.0f}x")
        
        del W, S, raw
        gc.collect()
    
    # Summary
    print("\n" + "=" * 64)
    print(" SUMMARY (at 99.99% energy preservation)")
    print("=" * 64)
    print(f"{'Tensor':<45} {'Shape':>15} {'Rank':>6} {'Ratio':>6}")
    print("-" * 80)
    
    total_dense = 0
    total_curve = 0
    for r in results:
        s = f"[{r['shape'][0]}x{r['shape'][1]}]"
        print(f"  {r['name']:<43} {s:>15} {r['rank']:>6} {r['ratio']:>5.1f}x")
        total_dense += r['dense_mb']
        total_curve += r['curve_mb']
    
    print(f"\n  Total dense (3 layers): {total_dense:.1f} MB")
    print(f"  Total curve (3 layers): {total_curve:.1f} MB")
    print(f"  Overall compression:    {total_dense/total_curve:.1f}x")
    print(f"  Projected 35 layers:    ~{total_dense/3*35:.0f} MB -> ~{total_curve/3*35:.0f} MB")

if __name__ == "__main__":
    measure_ranks("training/gemma-4-E2B-it-Q4_K_M.gguf")
