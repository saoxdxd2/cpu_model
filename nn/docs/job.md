# NCA — Component Registry (Job Status)

> **Status**: **GROUNDED PARAMETER CONTRACT (v27.0)**
> **Goal**: Comprehensive API and Logic Mapping.

| Component | Responsibility | Performance Target | Status |
| :--- | :--- | :--- | :--- |
| **MultimodalEngine** | Orchestrates Importance -> Spectral -> Expert ACT loop. | > 300 tok/s | **STABLE** |
| **ImportanceClassifier** | Saliency & Novelty detection for online grounding. | < 20 us / call | **SATURATED** |
| **SpectralLogic** | Kronecker-RLS domain correction and refinement. | < 100 us / call | **SATURATED** |
| **HashedRouter** | Rank-16 VNNI LSH router for Expert selection. | < 50 us / call | **SATURATED** |
| **FWHT** | Fast Walsh-Hadamard Transform for spectral mapping. | < 10 us / step | **PHYSICAL LIMIT** |
| **VNNI Kernels** | Low-level Rank-16 saturated linalg. | ~1 cycle / FMA | **PHYSICAL LIMIT** |

## 1. Top-Level API (`MultimodalEngine`)
- `step(text_in, img_in, out)`: Executes importance-gated grounded logic.
- `reset_state()`: Clears persistent RLS and momentum buffers.

## 2. Importance Contract (`importance.hpp`)
- `classify(x, state, pred)`: Returns `ImportanceDecision` (Fact/Novelty/ACT cycles).

## 3. Spectral Contract (`spectral_logic.hpp`)
- `spectral_logic_step(...)`: Refines state using RLS-based spectral correction.
