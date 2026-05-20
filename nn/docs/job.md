# NCA Function Registry & API Contract (job.md)

This document tracks all built and verified mathematical functions within the NCA project.
Always check this registry to reuse functions before writing new ones.

## `nca::math` (Core Dispatcher API)

### 1. `rmsnorm`
- **Header**: `core/normalization.hpp`
- **Signature**: `void rmsnorm(std::span<float> out, std::span<const float> in, std::span<const float> weight, float eps = 1e-5f)`
- **Description**: Applies Root Mean Square Normalization. 
- **Backends**: Dynamically routes to AVX-512, AVX2, or Scalar.

### 2. `silu`
- **Header**: `core/activations.hpp`
- **Signature**: `void silu(std::span<float> data)`
- **Description**: Applies Swish (SiLU) Activation in-place.
- **Backends**: Dynamically routes to AVX-512, AVX2, or Scalar.

## `nca::simd::avx512` & `avx2` (Inline Math)
- **Headers**: `core/simd/avx512_math.hpp` & `core/simd/avx2_math.hpp`
- **Description**: Contains branchless `__m512` and `__m256` inline math approximations like `exp_ps()` and `silu_ps()`. Use these natively inside specialized loops to avoid duplicating algorithm logic.

## `nca::simd::avx512` (AVX-512 Math Kernels)
*Compiled with `/arch:AVX512`. Safe to link, only call if CPUID allows.*

### 1. `rmsnorm`
- **Header**: `core/simd/avx512_kernels.hpp`
- **Optimization**: 4x unrolled, `_mm_prefetch`, Newton-Raphson `_mm512_rsqrt14_ps`, hardware mask tail.

## `nca::simd::avx2` (AVX2 Math Kernels)
*Compiled with `/arch:AVX2`.*

### 1. `rmsnorm`
- **Header**: `core/simd/avx2_kernels.hpp`
- **Optimization**: 4x unrolled, `_mm_prefetch`, Newton-Raphson `_mm256_rsqrt_ps`, masked tail fallback.

## `nca::simd` (Hardware Dispatch)
### 1. `detect`
- **Header**: `core/simd/dispatch.hpp`
- **Returns**: `CPUFeatures` struct containing cached CPUID results (AVX2, AVX-512, VNNI).

### 2. `best_backend`
- **Header**: `core/simd/dispatch.hpp`
- **Returns**: `Backend` enum (`AVX512`, `AVX2`, `SCALAR`).
