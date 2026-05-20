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

## What is SLOW (Avoid at all costs)

1. **Object-Oriented Abstractions in the Hot Loop**
   - Virtual functions, `std::function`, and heavy wrapper objects completely break auto-vectorization and ruin cycle predictability.

2. **Unaligned Memory Loads (`_mm512_loadu_ps`)**
   - While modern CPUs handle unaligned loads gracefully, they still incur a 1-3 cycle penalty when crossing cache line boundaries. Always design memory layouts to hit exactly 64 bytes.

3. **Optimizing Memory-Bound Operations**
   - If an operation takes less than 1 cycle per element (like our RMSNorm at 0.87 cycles/elem), it is bottlenecked by the physical DDR4 RAM speed. Do not try to optimize the math further. Instead, **fuse** the operation into a previous loop to save a memory read/write trip entirely.
