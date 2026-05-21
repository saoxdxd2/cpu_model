# NCA — Nano-Core Architecture Engineering Directives

This document defines the foundational mandates for development within this repository. Adherence to these principles is mandatory for all code contributions.

## 1. Core Philosophy: Complexity Destruction
- **Geometric over Relational**: Think about data as N-dimensional shapes. Transform loops to maximize rank-k updates and register reuse.
- **Algebraic Substitution**: Replace transcendental functions (log, exp) with geometric proxies (Gini impurity, minimax polynomials) to stay in the SIMD pipe.
- **Complexity Destruction**: If a task requires an $O(N \log N)$ sort, replace it with an $O(N)$ quick-select. If it requires 3 memory passes, algebraically reorder it into 1.

## 2. Safety & Modern C++ Standards (C++20)
- **Smart Pointers**: No manual `new`/`delete`. Use `aligned_unique_ptr` for SIMD safety.
- **STL First**: Use `std::nth_element` for Top-K and `std::sort` for locality preservation.
- **Dense Source**: Favor condensed, powerful logic over verbose boilerplate.

## 3. Heavy SIMD Mandates (AVX-512 & VNNI)
- **Log-Space Scaling**: Fuse weight and activation scales at the exponent level using `_mm512_scalef_ps`.
- **Quad-Row GEMV**: Process 4 rows simultaneously to reduce activation load traffic by 75%.
- **Branchless Decisioning**: Use bitwise blending and conditional arithmetic to eliminate ACT gating branches.

## 4. Hardware-Software Co-Design
- **L1 Residency**: "Pre-program" all loops to process 24-28KB tiles. Data MUST NOT exit the silicon until the block is fully resolved.
- **Atomic Jump Cache**: Use the `Dispatcher` to bind function pointers on first call for zero-cost runtime backend switching.

## 5. Development Workflow
- **AMI First**: Every math transformation must be verified via the Atomic Mathematical Isolation profiler.
- **PoC Adoption**: New concepts are adopted only if they pass AMI and show measurable cycle reduction.
