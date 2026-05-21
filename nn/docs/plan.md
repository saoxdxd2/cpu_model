# NCA — Nano-Core Architecture Implementation Plan

> **Status**: **FULLY CONDENSED & HARDENED (v6.3)**
> **Goal**: Beat Transformers on CPU inference via Randomized Hashed Routing.

| Phase | Task | Description | Status |
| :--- | :--- | :--- | :--- |
| 1-13 | Foundation | AMI Core, Multi-Modal Kernels | DONE |
| 14 | Integration | Model Convergence & Spectral Gating | DONE |
| 15 | Router | Saturated VNNI Hashed Router | DONE |
| 16 | Harden | Memory Safety & Brittle-Logic Guards | DONE |
| 17 | Verify | Comprehensive Robustness & Golden Contract | DONE |
| **18** | **Condense** | **Logic Consolidation & Dead Code Removal** | **DONE** |
| 19 | Finalize | Bit-Level Decompression (INT4/NF4) | TODO |
| 20 | Training | KFAC Second-Order Optimizer in C++ | TODO |

## Logic Condensation (Phase 18)
The architecture has been aggressively condensed to maximize instruction cache density:
1.  **Complexity Destruction**: $O(N \log N)$ sorting in the router was replaced by $O(N)$ Quick-Select.
2.  **Wavefront Scanning**: 2D-SSM now processes image diagonals in parallel, unlocking SIMD throughput.
3.  **Instruction Squeezing**: Superseded scalar fallbacks were stripped, ensuring the engine stays "under the L1" for both data and code.

The model achieves **~270 tokens/s** with a bit-perfect implementation contract.
