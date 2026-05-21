# NCA — Component Registry (Job Status)

> **Status**: **STABLE PARAMETER CONTRACT (v6.1)**
> **Goal**: Comprehensive API and Logic Mapping.

| Component | Responsibility | Performance Target | Status |
| :--- | :--- | :--- | :--- |
| **MultimodalEngine** | Orchestrates Vision -> Logic recursive ACT loop. | > 100 tok/s | **STABLE** |
| **HashedRouter** | Rank-16 VNNI LSH router for Expert selection. | < 50 us / call | **SATURATED** |
| **LatentAdapter** | Cross-modal projection (Vision -> Logic). | < 200 us / call | **SATURATED** |
| **SpectralPruner** | Entropy-based spatial gating (E-AdaPrune). | < 300 us / scan | **SATURATED** |
| **Backbone Stack** | Gated Linear RNN + State-Space Modeling. | < 10 us / step | **SATURATED** |
| **VNNI Kernels** | Low-level Rank-16 saturated linalg. | ~1 cycle / FMA | **PHYSICAL LIMIT** |

## 1. Top-Level API (`MultimodalEngine`)
- `step(text_in, img_in, out)`: Executes the complete neural circuit.
- `load_weights()`: (Phase 17) Loads .safetensors into persistent anchors.

## 2. Linalg Contract (`mx_linear.hpp`)
- `mx_rank16_dot`: Main saturated workhorse (1 Activation Load / 16 FMA).
- `mx_gemv`: Block-quantized matrix-vector multiplication.

## 3. Router Contract (`hashed_router.hpp`)
- `route(x, indices)`: Selects Top-K experts from pool using LSH.
