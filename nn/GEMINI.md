# NCA — Nano-Core Architecture Engineering Directives

This document defines the foundational mandates for development within this repository. Adherence to these principles is mandatory for all code contributions.

## 1. Core Philosophy: Maximum Optimization
- **Performance over Simplicity**: Squeeze every ounce of performance out of the silicon. Avoid standard pedagogical constructs if they incur overhead.
- **Complex, Condensed Code**: Favor efficiency and density over high-level abstractions.
- **Sequential over Relational**: Favor flat C-array sequential indexing and pointer arithmetic over complex tree traversals or object-oriented abstractions to maximize L1 cache utilization.

## 2. Safety & Modern C++ Standards (C++20)
- **Smart Pointers**: **Stop using manual new/delete**. Use `std::unique_ptr` and `std::shared_ptr` for ownership and allocation. Use custom deleters for aligned memory (`_aligned_free`).
- **Type Deduction**: Use `auto` for complex type names (iterators, lambdas) to improve readability.
- **STL First**: Embrace the STL for containers and algorithms. Use `std::unordered_map` or `std::unordered_set` to reduce complexity (e.g., $O(N^2) \to O(N)$).
- **Metaprogramming**: Use `constexpr`, `if constexpr`, and template specialization for zero-cost dispatch.
- **Safety**: Use `std::span` for robust boundary passing.
- **Guiding the Compiler**: Use `[[likely]]` and `[[unlikely]]` to guide branch prediction.

## 3. Heavy SIMD Mandates (AVX-512 & AVX2)
- **Branchless Programming**: inside critical paths, use `_mm512_mask_blend_ps`, float masking, or bitwise math instead of `if/else`.
- **Aggressive Unrolling**: Use 4x, 8x, or 16x unrolling to hide FMA latency and maximize ILP. Apply manual unrolling to `for` loops in hot paths.
- **Hardware Approximations**: Use `_mm512_rsqrt14_ps`, `_mm512_rcp14_ps` followed by Newton-Raphson. Avoid `std::sqrt`, `std::exp`, or scalar division.
- **Branchless Tails**: Process terminal blocks using `__mmask16` and masked operations to avoid scalar remainder loops.
- **VNNI first**: All dense projections and MLPs must target AVX-512 VNNI using MXINT8/MXUINT8 (E8M0) quantization.

## 4. Low-Level Memory & Cache Strategy
- **Pointer Arithmetic**: Prefer raw pointer arithmetic (`p += stride`) over indexed access.
- **Zero Aliasing**: All buffer arguments must be tagged with `__restrict`.
- **Software Prefetching**: Manually use `_mm_prefetch` to hide DDR4 latency.
- **Zero-Cost Views**: Use lightweight C-style struct views for tensor data to avoid RAII/container overhead in hot loops.

## 5. Atomic Mathematical Isolation (AMI)
- **Test-First**: Every mathematical concept must be written and validated in isolation.
- **Equivalence**: Compare SIMD kernels against scalar fallbacks via hash matching for bit-perfect (or near-perfect for transcendentals) parity.
- **Benchmarks**: New kernels are only adopted if they show a measurable performance improvement and pass AMI.

## 6. Development Workflow
- **Network Branching Optimization**: Minimize computational waste by ensuring efficient data flow between Nano-Cores (Vision, Logic, Action).
- **Reduce Duplication**: Reuse verified `AMI` primitives.
- **PoC Rule**: New concepts must be verified first. Do not merge half-finished ideas.
