# NCA Function Registry & API Contract (job.md)

This document tracks all built and verified mathematical functions within the NCA project.
Always check this registry to reuse functions before writing new ones.

## `nca::math` (Core Dispatcher API)

### 1. `rmsnorm`
- **Header**: `core/normalization.hpp`
- **Signature**: `void rmsnorm(std::span<float> out, std::span<const float> in, std::span<const float> weight, float eps = 1e-5f)`
- **Description**: Applies Root Mean Square Normalization. 
- **Backends**: Dynamically routes to AVX-512, AVX2, or Scalar.
- **Performance**: ~0.23 cycles/element (memory-bound)

### 2. `silu`
- **Header**: `core/activations.hpp`
- **Signature**: `void silu(std::span<float> data)`
- **Description**: Applies Swish (SiLU) Activation in-place.
- **Backends**: Dynamically routes to AVX-512, AVX2, or Scalar.
- **Performance**: ~0.53 cycles/element

## `nca::layers` (Backbone Layers)

### 3. `glr_step`
- **Header**: `core/layers/glr.hpp`
- **Signature**: `void glr_step(float* h, const float* alpha, const float* beta, const float* x, size_t d_size)`
- **Description**: Gated Linear Recurrence temporal step: `h = alpha*h + beta*x`.
- **Backends**: AVX-512 (4x unrolled + CachePolicy), AVX2, Scalar.
- **Performance**: ~0.19 cycles/element (memory-bound, L2_STREAM policy)
- **Tricks**: `constexpr if` CachePolicy, branchless masked tail via `__mmask16`.

### 4. `ssm_step`
- **Header**: `core/layers/ssm.hpp`
- **Signature**: `void ssm_step(float* h, const float* A, const float* B, const float* C, const float* x, float* y, SSMConfig cfg)`
- **Description**: Selective State-Space Model (Mamba-style) temporal step. `h = A*h + B*x, y = C*h`.
- **Backends**: AVX-512 (4x unrolled, d_state=16 fastpath), Scalar.
- **Performance**: ~5.66 cycles/element (DDR4-bound: 1.1MB working set)
- **Tricks**: Hoisted B/C broadcasts, `constexpr` DDR4_NT prefetch policy.

### 5. `mx_fused_ssm_silu_quantize_step`
- **Header**: `core/layers/ssm.hpp`
- **Signature**: `void mx_fused_ssm_silu_quantize_step(...)`
- **Description**: Fused SSM memory scan + SiLU activation + MX block quantization.
- **Backends**: AVX-512.
- **Performance**: ~5.47 cycles/element.
- **Tricks**: Fused loop locally buffers `y` values to avoid flushing 32KB of intermediate arrays to L2/L3 cache before quantizing.

### 5. `sla_step`
- **Header**: `core/layers/sla.hpp`
- **Signature**: `void sla_step(const float* q, const float* k_cache, const float* v_cache, float* out, float* scores, SLAConfig cfg)`
- **Description**: Sparse Local Attention sliding window (W=256). Autoregressive single-query step.
- **Backends**: AVX-512 (Branchless online softmax, fully unrolled), Scalar.
- **Performance**: ~0.88 cycles/element (L2_STREAM policy)
- **Tricks**: 3-phase fused pipeline, branchless softmax, and L2 prefetching.

### 7. `gated_mlp_step`
- **Header**: `core/layers/mlp.hpp`
- **Signature**: `void gated_mlp_step(...)`
- **Description**: Computes `y = W_down * silu(W_gate * x) * (W_up * x)` using block-quantized weights.
- **Backends**: AVX-512 (VNNI), Scalar.
- **Performance**: ~3695 cycles/element (Full 8192x8192 step).
- **Tricks**: Uses `mx_gemv` under the hood. Avoids L1/L2 cache pollution by streaming 192MB of weights via Non-Temporal (NT) loads, keeping activations pinned in L1.

### 8. `mx_gemv`
- **Header**: `core/linalg/mx_linear.hpp`
- **Signature**: `void mx_gemv(const MXINT8Tensor& W, const MXUINT8Tensor& x, float* y, size_t rows, size_t cols)`
- **Description**: Matrix-Vector multiplication using VNNI INT8 block quantization.
- **Backends**: AVX-512 (VNNI), Scalar.
- **Performance**: ~1379 cycles/element.
- **Tricks**: Defers all horizontal reductions to the end of the dot product (`_mm512_reduce_add_ps` is called only once per row). Uses `_mm512_stream_load_si512` for weight fetching to bypass cache..

### 7. `halting_step`
- **Header**: `core/layers/halting.hpp`
- **Signature**: `float halting_step(...)`
- **Description**: Adaptive Computation Time (ACT) Halting Gate classifier.
- **Backends**: AVX-512 VNNI `mx_dot`.
- **Performance**: ~0.50 cycles/element

## `nca::linalg` (MX Block-Quantized Linear Algebra)

### 5. `mx_quantize_w`
- **Header**: `core/linalg/mx_linear.hpp`
- **Signature**: `void mx_quantize_w(const float*, MXINT8Tensor&)`
- **Description**: Offline weight quantization to MXINT8 format. Runs once at model load.
- **Performance**: ~8.68 cycles/element (scalar, not perf-critical)

### 6. `mx_quantize_x`
- **Header**: `core/linalg/mx_linear.hpp`
- **Signature**: `void mx_quantize_x(const float*, MXUINT8Tensor&)`
- **Description**: Dynamic activation quantization to MXUINT8 for VNNI.
- **Backends**: AVX-512 vectorized, Scalar fallback.
- **Performance**: ~0.25 cycles/element

### 7. `mx_fused_silu_quantize_x`
- **Header**: `core/linalg/mx_linear.hpp`
- **Signature**: `void mx_fused_silu_quantize_x(const float*, MXUINT8Tensor&)`
- **Description**: **Kernel Fusion**: SiLU + Quantize in a single AVX-512 pass. Eliminates the DDR4 round-trip between separate SiLU and Quantize calls.
- **Backends**: AVX-512 fused, Scalar fallback.
- **Performance**: ~0.77 cycles/element

### 8. `mx_dot`
- **Header**: `core/linalg/mx_linear.hpp`
- **Signature**: `float mx_dot(const MXINT8Tensor& w, const MXUINT8Tensor& x)`
- **Description**: MX block-quantized dot product using VNNI `VPDPBUSD`.
- **Backends**: AVX-512 VNNI only (throws on unsupported hardware).
- **Performance**: ~0.10 cycles/element

## `nca::simd` (Infrastructure)

### 9. `detect` / `best_backend`
- **Header**: `core/simd/dispatch.hpp`
- **Description**: CPUID detection singleton. Cached. Routes all kernels.

### 10. `CachePolicy<WorkingSetBytes>`
- **Header**: `core/simd/cache_policy.hpp`
- **Description**: Compile-time constexpr decision tree. Given a working set size in bytes, computes:
  - `strategy`: L1_HOT / L2_STREAM / DDR4_NT
  - `prefetch_dist`: prefetch distance in cache lines
  - `use_nt_stores`: whether to use `_mm512_stream_ps`
  - `tile_lines`: how many cache lines fit in half L1
- **Usage**: `using Policy = CachePolicy<4 * D * 4>;` then `if constexpr (Policy::use_nt_stores) ...`

### 11. `tail_mask(size_t remaining)`
- **Header**: `core/simd/cache_policy.hpp`
- **Description**: Branchless `__mmask16` generation for partial-vector tail handling. Replaces all `if (remaining > 0)` scalar loops with masked SIMD operations.

## `nca::simd::avx512` & `avx2` (Inline Math)
- **Headers**: `core/simd/avx512_math.hpp` & `core/simd/avx2_math.hpp`
- **Description**: Branchless `__m512` and `__m256` inline math: `exp_ps()`, `silu_ps()`.

## `nca::vision` (Vision Layers)

### 12. `dwconv2d_3x3`
- **Header**: `core/vision/scanner.hpp`
- **Signature**: `void dwconv2d_3x3(...)`
- **Description**: NHWC 3x3 Depthwise Convolution over a 2D feature map.
- **Backends**: AVX-512 (fully unrolled 9 FMAs per channel block), Scalar.
- **Performance**: ~0.68 cycles/element.
- **Tricks**: Processes multiple channels simultaneously via SIMD over the `C` dimension (NHWC layout).

### 13. `ssm2d_scan`
- **Header**: `core/vision/scanner.hpp`
- **Signature**: `void ssm2d_scan(...)`
- **Description**: 2D flattened Selective State-Space Model scan for global receptive field.
- **Backends**: Wraps AVX-512 `ssm_step`.
- **Performance**: ~6.2 cycles/element.

## Auto-Profiler Infrastructure
- **Python**: `tests/autogenerate.py` — zero-argument, discovers core/ headers by itself.
- **C++ TMP**: `tests/autotest.hpp` — `StripRestrict<T>` + `ArgGen<T>` + `BenchmarkRunner<Func>`.
- **CMake**: `GLOB_RECURSE CONFIGURE_DEPENDS` — re-triggers on any `.hpp` change.
