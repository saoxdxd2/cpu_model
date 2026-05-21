# NCA — Extreme Optimization Advice & Custom Rules

This living document tracks our hard-won discoveries regarding extreme hardware optimization on x86 CPUs (specifically AVX-512 and Ice Lake architectures). Use these exact principles when writing future layers.

## What is FAST (Adopt these immediately)

1. **Pure C-Style Arrays & 64-Byte Alignment**
   - **Why**: C++ `std::vector` lacks explicit alignment guarantees and adds initialization overhead. By using `_aligned_malloc(size, 64)`, we perfectly align memory boundaries for 512-bit AVX registers.
   - **Result**: Allowed our `mx_dot` VNNI kernel to use `_mm512_load_si512` directly, achieving **10 elements processed per single CPU cycle** (0.09 cycles/elem).

2. **Branchless Programming**
   - **Why**: CPU branch predictors violently stall execution pipelines upon failure. 
   - **Rule**: Never use `if/else` inside hot loops. Use bitwise masks (`_mm512_mask_blend_ps`) or Minimax Polynomial algorithms instead.

3. **Hardware Approximations over Standard Library**
   - **Why**: `std::exp` and `std::sqrt` are extremely slow and require heavy scalar transcendentals.
   - **Rule**: Use hardware root approximations (`_mm512_rsqrt14_ps`) followed by Newton-Raphson iterations. Use fused-multiply-add polynomial approximations paired with `_mm512_scalef_ps` for exponents.

4. **Non-Temporal Memory Streams (`_mm512_stream_ps`)**
   - **Why**: Standard stores (`_mm512_store_ps`) force the CPU to read the target memory block into the L1 cache before overwriting it ("read-for-ownership"). `stream_ps` bypasses the cache entirely and streams data directly to the RAM chips, saving massive memory bandwidth on large tensor arrays.
   - **Rule**: If an output array is exactly 64-byte aligned and won't be read again immediately, use `_mm512_stream_ps`.

5. **`__restrict` Pointers**
   - **Why**: C++ compilers cannot aggressively unroll loops if they suspect two pointers might overlap in memory.
   - **Rule**: Always tag API arguments as `float* __restrict` to guarantee zero aliasing to the compiler.

6. **Kernel Fusion (Layer Fusion)**
   - **Why**: On CPU architectures, sequential compute-bound functions (like SiLU followed by Quantization) kill performance if they write to RAM only to read it immediately.
   - **Rule**: Fuse intermediate operations deeply inside the loop at the L1 register level. Compute SiLU within the `__m512` register and immediately execute the `max_abs` and downcast to `uint8` *before* the first memory store is emitted. Eliminating memory hops is the only way to beat DDR4 latency.

7. **Constexpr CachePolicy Decision Tree**
   - **Why**: Different kernels have different working set sizes. A D=2048 GLR fits L1 (32KB); a D=8192 GLR fits L2 (256KB); an SSM exceeds L2 (1MB+). Each needs a different strategy (no prefetch vs. prefetch vs. NT stores). Deciding this at runtime wastes branches.
   - **Rule**: Use `CachePolicy<WorkingSetBytes>` from `cache_policy.hpp`. It computes `strategy`, `prefetch_dist`, and `use_nt_stores` at compile time. Use `if constexpr` to emit only the optimal code path. Zero runtime branches for policy decisions.
   - **Result**: GLR went from 1.65 → 0.19 cycles/element by switching from hardcoded NT stores to CachePolicy-selected L2_STREAM mode.

8. **Branchless Tail Masking (`__mmask16`)**
   - **Why**: The tail of a SIMD loop (when `remaining < vector_width`) is traditionally handled with a scalar `for` loop. This causes a branch mispredict and scalar pipeline stall.
   - **Rule**: Use `tail_mask(remaining)` to generate a `__mmask16` and process the tail with `_mm512_maskz_loadu_ps` / `_mm512_mask_storeu_ps`. The mask zeros out invalid lanes. No branch, no scalar code.

## What is SLOW (Avoid at all costs)

1. **Object-Oriented Abstractions in the Hot Loop**
   - Virtual functions, `std::function`, and heavy wrapper objects completely break auto-vectorization and ruin cycle predictability.

2. **Unaligned Memory Loads (`_mm512_loadu_ps`)**
   - While modern CPUs handle unaligned loads gracefully, they still incur a 1-3 cycle penalty when crossing cache line boundaries. Always design memory layouts to hit exactly 64 bytes.

3. **Optimizing Memory-Bound Operations**
   - If an operation takes less than 1 cycle per element (like our RMSNorm at 0.23 cycles/elem), it is bottlenecked by the physical DDR4 RAM speed. Do not try to optimize the math further. Instead, **fuse** the operation into a previous loop to save a memory read/write trip entirely.

4. **NT Stores on Read-Modify-Write Arrays**
   - `_mm512_stream_ps` bypasses the CPU cache. If you stream-store `h[]` and then immediately read it back (e.g., to compute `y = C * h`), the data is NOT in L1 — it was flushed straight to DDR4. The subsequent read causes a full cache miss.
   - **Rule**: Only use NT stores for arrays that are **written once and not re-read** in the same kernel invocation (e.g., GLR output in a separate write-only buffer). For read-modify-write arrays (like SSM's `h[]`), always use regular `_mm512_storeu_ps`.
