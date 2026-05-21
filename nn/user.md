# User Coding Style Preferences & Engineering Directives

The core directive is **"Maximum Optimization"** and **"Complex, Condensed"** C++ code.

## Key Directives:

### 1. Extreme Performance over Simplicity
- The code must squeeze every ounce of performance out of the silicon.
- Avoid standard pedagogical constructs (like simple `for` loops with array indices).
- Favor flat C-array sequential indexing (like the `RoutePlanner`) over complex tree traversals to maximize L1 cache utilization.

### 2. Heavy SIMD Optimization (AVX-512 & AVX2)
- **Branchless Programming**: Avoid `if/else` logic inside critical paths. Compute all paths and blend via bitwise math, intrinsic select operations (`_mm512_mask_blend_ps`), or float masking (e.g., `exceeds * p_clamped + (1 - exceeds) * p`). Branch prediction misses cost massive pipeline stalls; branchless logic guarantees deterministic throughput.
- **Loop Unrolling**: Unroll loops aggressively (e.g., 4x or 8x) to break data dependency chains, hide FMA latencies, and increase ILP (Instruction-Level Parallelism).
- **Hardware Approximations**: Avoid scalar divisions or transcendentals (e.g., `std::sqrt`). Use hardware approximations (`_mm512_rsqrt14_ps`) followed by manual Newton-Raphson iterations to restore IEEE precision in vector registers.
- **Branchless Tails**: Use hardware masking (`__mmask16`, `_mm512_mask_storeu_ps`) or branchless tail masks (using `_bzhi_u32` or bit shifts) for terminal blocks instead of scalar remainder loops.
- **VNNI Kernels**: Use INT8 quantization (E8M0 contract) with AVX-512 VNNI (`_mm512_dpbusd_epi32`) for maximum throughput in MLPs and projections.

### 3. Low-level Memory Management & Cache Strategy
- **Pointer Arithmetic**: Use raw pointer arithmetic (`p += 64`) instead of indexed array access (`arr[i]`).
- **Zero Aliasing**: Guarantee zero aliasing with `__restrict` pointers on all buffers.
- **Software Prefetching**: Manually hide memory latency via software prefetching (`_mm_prefetch(..., _MM_HINT_T0)`) ahead of the CPU execution.
- **Compile-Time Cache Policy**: Compute the optimal tiling strategy, prefetch distance, and store policy (e.g., non-temporal streaming `_mm512_stream_ps` vs standard stores) at compile time via `constexpr` to avoid runtime branches.
- **Zero-Cost Views**: Avoid RAII container overhead in hot loops (e.g., copying tensor structs). Use lightweight C-style struct views to borrow pointers.

### 4. Modern C++20 Usage
- **Metaprogramming**: Use `constexpr`, `if constexpr`, and template specialization for zero-cost dispatch and configuration.
- **Zero-Cost Logging**: Use `std::source_location` to capture file, line, and function names at compile time, eliminating the need for bulky macros.
- **Bounds Checking**: Use `std::span` for robust boundary passing from higher-level APIs.
- **Compiler Hints**: Use `[[likely]]` and `[[unlikely]]` to guide the compiler's branch predictor.
- Strict adherence to the C++20 standard.

### 5. Atomic Mathematical Isolation (AMI) & Testing
- Every mathematical concept must be written and tested via the AMI framework.
- The auto-profiler compares scalar fallback output against the AVX-512 kernel via hash matching to ensure bit-perfect equivalence (or expected FP approximation error for transcendentals).
- Only once a kernel passes the correctness check and achieves maximum speed is it moved to production.

### 6. Proof-of-Concept Adoption Rule
- Whenever a new, interesting concept is proposed, we must **test and verify it first**.
- If it works perfectly and is fully optimized, we adopt it.
- If it shows promise but requires significant further improvement, we **do not adopt it** into the core framework until we finish the current task. Never leave half-finished "interesting" ideas in the codebase.
