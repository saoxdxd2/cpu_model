# NCA — Nano-Core Architecture Implementation Plan

> **Status**: **STABLE PARAMETER CONTRACT (v6.1)**
> **Goal**: Beat Transformers on CPU inference via Randomized Hashed Routing.

| Phase | Task | Description | Status |
| :--- | :--- | :--- | :--- |
| 1-13 | Foundation | AMI Core, Multi-Modal Kernels | DONE |
| 14 | Integration | Model Convergence & Spectral Gating | DONE |
| **15** | **Router** | **Saturated VNNI Hashed Router** | **DONE** |
| 16 | Stabilize | Fixed-Seed Contract & Deterministic Testing | DONE |
| 17 | Finalize | Bit-Level Decompression (INT4/NF4) | TODO |
| 18 | Training | KFAC Second-Order Optimizer in C++ | TODO |

## Parameter Contract (Phase 16)
The architecture is now mathematically stable and Reproducible:
1.  **Saturated Router**: `HashedRouter` now uses real Rank-16 VNNI kernels for expert selection.
2.  **Deterministic Contract**: Weights are initialized via a **Fixed Seed** to ensure output reproducibility.
3.  **Golden Reference**: `test_final.cpp` validates the model against a deterministic FNV-1a hash.
