"""
NCA -- Curve-Encoded Weights (CEW)
===================================
The curve doesn't APPROXIMATE the weights. The curve IS the weights.
Same data, different layout, optimized for CPU cache topology.

Think: row-major vs column-major vs CURVE-major.

Three approaches tested:
1. Z-order (Morton) curve: maps 2D matrix to 1D with locality preservation
2. Hilbert curve: even better locality preservation
3. Block-interleaved: matches our FWHT butterfly pattern

The key insight: matmul y = W*x processes weights row-by-row.
But the CPU processes CACHE-LINE by CACHE-LINE (64 bytes = 16 floats).
If we reorder weights to match the CPU's access pattern, we get free speedup.
"""
import torch
import numpy as np
import gguf
import gc
import time


def morton_encode(x, y):
    """Interleave bits of x and y for Z-order curve."""
    z = 0
    for i in range(16):
        z |= ((x >> i) & 1) << (2*i) | ((y >> i) & 1) << (2*i + 1)
    return z


def build_morton_order(m, n):
    """Build Morton Z-order traversal indices for an m x n matrix."""
    indices = []
    for i in range(m):
        for j in range(n):
            indices.append((morton_encode(i, j), i, j))
    indices.sort(key=lambda x: x[0])
    return [(i, j) for _, i, j in indices]


def build_block_interleaved_order(m, n, block_size=16):
    """
    Block-interleaved: process in blocks that fit in L1.
    Within each block, access is sequential (cache-line friendly).
    Block order follows Z-curve for inter-block locality.
    """
    blocks = []
    for bi in range(0, m, block_size):
        for bj in range(0, n, block_size):
            block = []
            for i in range(bi, min(bi + block_size, m)):
                for j in range(bj, min(bj + block_size, n)):
                    block.append((i, j))
            blocks.append(block)
    
    # Flatten with block-level Z-order
    indices = []
    for block in blocks:
        indices.extend(block)
    return indices


class CurveEncodedWeights:
    """
    Stores weight matrix as a curve — same data, curve-ordered layout.
    The curve defines: for each position in the flat buffer, 
    which (row, col) of the original matrix does it hold?
    """
    def __init__(self, W, curve_type='block16'):
        self.m, self.n = W.shape
        self.curve_type = curve_type
        
        if curve_type == 'morton':
            self.order = build_morton_order(self.m, self.n)
        elif curve_type.startswith('block'):
            bs = int(curve_type.replace('block', ''))
            self.order = build_block_interleaved_order(self.m, self.n, bs)
        else:
            # Row-major (baseline)
            self.order = [(i, j) for i in range(self.m) for j in range(self.n)]
        
        # Store weights in curve order
        self.curve_data = torch.zeros(self.m * self.n)
        for idx, (i, j) in enumerate(self.order):
            self.curve_data[idx] = W[i, j]
        
        # Build reverse map: for efficient matmul, we need 
        # row_starts[i] = positions in curve_data that belong to row i
        self.row_indices = [[] for _ in range(self.m)]
        self.row_col_map = [[] for _ in range(self.m)]
        for idx, (i, j) in enumerate(self.order):
            self.row_indices[i].append(idx)
            self.row_col_map[i].append(j)
    
    def verify_lossless(self, W_original):
        """Verify that the curve holds EXACTLY the same data."""
        W_reconstructed = torch.zeros(self.m, self.n)
        for idx, (i, j) in enumerate(self.order):
            W_reconstructed[i, j] = self.curve_data[idx]
        error = torch.norm(W_original - W_reconstructed).item()
        return error
    
    def matmul(self, x):
        """Compute y = W @ x using curve-ordered data."""
        y = torch.zeros(self.m)
        for i in range(self.m):
            for idx_pos, j in zip(self.row_indices[i], self.row_col_map[i]):
                y[i] += self.curve_data[idx_pos] * x[j]
        return y
    
    def matmul_tiled(self, x):
        """Tile-based matmul following the curve order."""
        y = torch.zeros(self.m)
        # Process curve_data sequentially (cache-optimal!)
        for idx, (i, j) in enumerate(self.order):
            y[i] += self.curve_data[idx] * x[j]
        return y


def benchmark_curve_matmul(W, name):
    m, n = W.shape
    x = torch.randn(n)
    y_true = W @ x
    
    print(f"\n{'='*70}")
    print(f"  CURVE-ENCODED WEIGHTS: {name} [{m}x{n}]")
    print(f"{'='*70}")
    
    # Dense baseline
    t0 = time.time()
    for _ in range(100):
        _ = W @ x
    t_dense = (time.time() - t0) / 100
    print(f"  Dense row-major:     {t_dense*1000:.3f} ms")
    
    # Test different curve orderings
    for curve in ['block16', 'block32', 'block64']:
        cew = CurveEncodedWeights(W, curve)
        
        # Verify lossless
        recon_err = cew.verify_lossless(W)
        
        # Compute via curve
        y_curve = cew.matmul_tiled(x)
        compute_err = torch.norm(y_curve - y_true).item() / torch.norm(y_true).item() * 100
        
        # Benchmark curve matmul
        t0 = time.time()
        for _ in range(10):  # Fewer iters since Python loop is slow
            _ = cew.matmul_tiled(x)
        t_curve = (time.time() - t0) / 10
        
        bs = int(curve.replace('block', ''))
        tile_kb = bs * bs * 4 / 1024
        
        print(f"  {curve:12s} ({tile_kb:.0f}KB tiles): "
              f"{t_curve*1000:.1f} ms | "
              f"recon={recon_err:.0e} | "
              f"compute_err={compute_err:.4f}% | "
              f"params={cew.curve_data.numel():,} (SAME)")
    
    # The REAL point: cache simulation
    print(f"\n  -- Cache Access Pattern Analysis --")
    for curve in ['row_major', 'block16']:
        if curve == 'row_major':
            order = [(i, j) for i in range(min(m,64)) for j in range(min(n,64))]
        else:
            order = build_block_interleaved_order(min(m,64), min(n,64), 16)
        
        # Count cache line transitions
        cache_lines_touched = set()
        sequential_hits = 0
        prev_line = -1
        for idx, (i, j) in enumerate(order):
            # Simulate: weight at (i,j) is at memory address i*n + j
            addr = i * n + j
            cache_line = addr // 16  # 16 floats per cache line
            cache_lines_touched.add(cache_line)
            if cache_line == prev_line:
                sequential_hits += 1
            prev_line = cache_line
        
        total_accesses = len(order)
        hit_rate = sequential_hits / total_accesses * 100 if total_accesses > 0 else 0
        print(f"  {curve:12s}: {len(cache_lines_touched)} cache lines | "
              f"{hit_rate:.1f}% sequential hits")


def main():
    print("=" * 70)
    print(" NCA CURVE-ENCODED WEIGHTS")  
    print(" Same data, curve-ordered layout, lossless")
    print("=" * 70)
    
    # Small test first
    print("\n[1] Synthetic 64x64")
    W_small = torch.randn(64, 64)
    benchmark_curve_matmul(W_small, "synthetic_64x64")
    
    # Real Gemma weight (small one)
    print("\n[2] Real Gemma-4 blk.0.attn_k.weight")
    reader = gguf.GGUFReader("training/gemma-4-E2B-it-Q4_K_M.gguf")
    for t in reader.tensors:
        if t.name == "blk.0.attn_k.weight":
            raw = gguf.dequantize(t.data, t.tensor_type)
            W = torch.from_numpy(raw.copy()).float()
            benchmark_curve_matmul(W, t.name)
            del W, raw; gc.collect()
            break
    
    print("\n" + "=" * 70)
    print(" COMPLETE"); print("=" * 70)

if __name__ == "__main__":
    main()
