# NCA — Extreme Optimization Advice & Custom Rules

This living document tracks our hard-won discoveries regarding extreme hardware optimization on x86 CPUs (specifically AVX-512 and Ice Lake architectures). Use these exact principles when writing future layers.

## What is FAST (Adopt these immediately)

1. **Register Reuse via Row Unrolling (Dual/Quad Row)**
   - **Why**: Loading the activation vector `x` into registers is expensive. By processing 2 or 4 rows of weights against the same `x`, you reuse the activation registers 2x-4x.
   - **Result**: Reduced L1-to-Register traffic by 75% in `mx_gemv`.

2. **Shuffle-to-Contiguous (The "Big Data" Shuffle)**
   - **Why**: Random access (Gathers) are pipeline killers. Physically copying data into a contiguous buffer via SIMD turns random access into a sequential scan that the hardware prefetcher loves.
   - **Rule**: If you have sparse active tokens, shuffle them into a contiguous "Active Buffer" once, then run sequential SIMD kernels.

3. **Atomic Jump Cache (Dynamic Dispatch)**
   - **Why**: Checking `best_backend()` inside a hot loop is a branch.
   - **Rule**: Use the `Dispatcher` class to bind a function pointer on the first call. Subsequent calls are a single direct jump.

4. **L1-Cache Tiling (The "Keep it under L1" Rule)**
   - **Why**: Intel Ice Lake L1d is exactly 32KB.
   - **Rule**: "Pre-program" your loops to process tiles of ~24-28KB. This ensures states (like SSM's `h`) never leave the high-speed silicon until the task is complete.

5. **`scalef_ps` for E8M0 Scaling**
   - **Why**: `2^x` is a native instruction (`_mm512_scalef_ps`). 
   - **Rule**: Never use a lookup table or `std::pow` for E8M0 scales. Use `scalef_ps` to apply the power-of-two multiplier branchlessly in the SIMD pipe.

6. **Smart Pointers with Aligned Deleters**
   - **Why**: RAII is safer, but standard `std::unique_ptr` doesn't know about `_aligned_free`.
   - **Rule**: Always use `aligned_unique_ptr<T[]>` from `core/simd/memory.hpp`.

## What is SLOW (Avoid at all costs)

1. **Horizontal Reductions in the Inner Loop**
   - `_mm512_reduce_add_ps` is an multi-instruction operation. Defer all reductions to the end of the row. 
   - **Tweak**: Use vertical unrolling to process many channels before reducing.

2. **Standard STL Containers in Hot Paths**
   - `std::vector` overhead (size checks, capacity checks) kills cycle-exact predictability. Use raw pointers or `std::span` borrowed from `aligned_unique_ptr`.
