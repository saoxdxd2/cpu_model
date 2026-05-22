# NCA — Extreme Optimization Advice & Custom Rules

## The Geometric Mandate (Phase 9 Discovery)

1. **Think Geometric, Not Relational**
   - **Row-by-Row is SLOW**: Loading the same activation vector for every row kills throughput.
   - **Block-by-Block is FAST**: Process 4, 8, or 16 rows at once (**Rank-K update**). Reuse the registers you just loaded. 
   - **Result**: We proved that `mx_quad_dot` (4 dots) is faster than `mx_dot` (1 dot) because it bypasses the load-unit bottleneck.

2. **Algebraic Substitution & Branchless Execution**
   - Replace transcendental functions with geometric proxies.
   - Use **Gini Impurity** or **Minimax Sigmoids** instead of Entropy/Log.
   - **Eliminate Branches**: Use conditional operators (ternary) to force the compiler into using `cmov` instructions. Unpredictable branches ruin instruction cache pipelines.

3. **Pre-Program the Loop (AVX2 Baseline)**
   - Don't let the compiler guess. "Pre-program" your tiling and unrolling based on the **32KB L1d** target. 
   - Ensure the working set of your fused kernel fits entirely in the silicon cache.
   - **Strict AVX2 Baseline**: The architecture assumes AVX2 is the absolute minimum hardware. **Do not write pure scalar fallback loops**; they pollute the instruction cache. Use intrinsics or write branchless C++ that the compiler natively auto-vectorizes.

4. **Shuffle, Don't Gather**
   - If you have sparse data, physically **SHUFFLE** it into a contiguous buffer once.
   - Then run your sequential SIMD kernels at full speed. Never use random-access Gathers in a hot loop.

5. **Atomic Jump Dispatch**
   - Use the `Dispatcher` to bind function pointers once. Any `if (backend == AVX512)` check inside a loop is a violation of the perfectly-pre-programmed principle. Rely on compile-time macros (`#if defined(__AVX512F__)`) or the `Dispatcher` to jump directly into the saturated silicon path.
