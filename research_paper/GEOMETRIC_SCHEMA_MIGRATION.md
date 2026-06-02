# CENTAUR Architectural Migration Plan: The Geometric Schema

**Date:** June 2026
**Target Architecture:** SIMD Stochastic Wavefront (AVX-512)
**Objective:** Deprecate continuous $O(N^2)$ Dense Logic Matrices in favor of the discrete, stochastic, 8-byte Geometric Wavefront, yielding massive latency acceleration and total immunity to noise diffusion.

---

## 1. Baseline vs. Target Comparison

| Metric | Legacy Baseline | Target Geometric Schema |
| :--- | :--- | :--- |
| **Execution Core** | FP32/INT8 Dense Matrices | AVX-512 Cache-Aligned Routing |
| **Inference Math** | Continuous $O(N^2)$ | Discrete Top-16 Wavefront $O(\text{active})$ |
| **Probability Spread** | Diffuses into Noise | Sharp, Top-16 DDQN Tracking |
| **Exploration / Temp** | Sequential scalar RNG | Vectorized AVX-512 Xorshift |
| **Training Paradigm** | Calculus / Gradient Descent | Reinforcement Learning / DDQN |
| **Memory Allocation** | Dynamic Heap (`std::vector`) | Static Arena Allocation |

---

## 2. Phase I: The Translation Pipeline
We update our weights processing to act as the **Transformer-to-Geometric Compiler**.

**Tasks:**
1. **Weight Normalization:** Convert extracted FP32 dense matrices into normalized probability distributions.
2. **Top-K Distillation:** Sort the continuous connections for each concept and isolate the top structural branches.
3. **Geometric Encoding:** Pack the data into the 8-byte struct format.
4. **Binary Serialization:** Export a raw C++ binary graph (`.geo`).

---

## 3. Phase II: The Execution Engine
The legacy engine is rebuilt to enforce absolute architectural minimization.

**Tasks:**
1. **The Geometric Router:** Implement a router that resolves states using intrinsic pointer-free arithmetic.
2. **Parallel Fuzziness:** Track parallel logical realities simultaneously in a single `__m512` register.
3. **Hardware Randomness (Xorshift):** Integrate the `SimdRandomState` logic directly into the wavefront hop, achieving Gumbel-Max probabilistic sampling without touching RAM.
4. **Eliminate Abstractions:** Remove `std::vector`, `std::shared_ptr`, and `std::span` in favor of raw buffers.

---

## 4. Phase III: Continuous-Discrete Unification
Transition routing and execution to strict boolean manifolds.

**Tasks:**
1. Implement the **Bipolar Phase-Collapse Router (BPCR)** to eliminate the $W_{up}$ matrix memory overhead.
2. Formulate **Binary Curve Tree (BCT)** logic to bypass FP32 multipliers using IEEE-754 XOR logic.
3. Enforce branchless execution to achieve topological hardware saturation.

---
**Status:** Executing Phase III.
