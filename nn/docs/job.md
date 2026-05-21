# NCA Function Registry & API Contract (job.md)

This document tracks all built and verified mathematical functions within the NCA project.
Always check this registry to reuse functions before writing new ones.

## `nca::math` (Core Dispatcher API)

### 1. `rmsnorm`
- **Signature**: `void rmsnorm(std::span<float> out, std::span<const float> in, std::span<const float> weight, float eps)`
- **Performance**: ~0.51 cycles/element.

### 2. `silu`
- **Signature**: `void silu(std::span<float> data)`
- **Performance**: ~2.5 cycles/element.

## `nca::layers` (Backbone Layers)

### 3. `glr_step`
- **Signature**: `void glr_step(float* h, const float* alpha, const float* beta, const float* x, size_t d_size)`
- **Performance**: **0.26 cycles/element** (Limit Saturated).

### 4. `ssm_step`
- **Signature**: `void ssm_step(float* h, const float* A, const float* B, const float* C, const float* x, float* y, SSMConfig cfg)`
- **Tricks**: **L1-Cache Tiling** (192-channel tiles) to stay under 32KB.

### 5. `mx_fused_ssm_silu_quantize_step`
- **Description**: **Horizontal Fusion**. Fuses SSM scan with Logic-Core quantization. 
- **Backends**: AVX-512 (Fused).

### 6. `gated_mlp_step`
- **Description**: Full SwiGLU MLP step using `mx_quad_dot` backend.

## `nca::linalg` (MX Block-Quantized Linear Algebra)

### 7. `mx_dot` / `mx_dual_dot` / `mx_quad_dot`
- **Description**: Dot products using VNNI `VPDPBUSD`.
- **Quad-Dot Trick**: Processes 4 weights rows against 1 shared activation vector. Reduces L1 traffic by 75%.
- **Performance**: ~0.9 cycles/dot (Shared L1).

### 8. `mx_gemv`
- **Signature**: `void mx_gemv(const MXINT8Tensor& W, const MXUINT8Tensor& x, float* y, size_t rows, size_t cols)`
- **Performance**: Optimized via **4-Row Unrolling**.

## `nca::execution` (Routing & Fusion)

### 9. `shuffle_active_tokens`
- **Description**: Physically reorders scattered active tokens into a contiguous SIMD buffer.
- **Trick**: Replaces slow Gather-access with fast Sequential Scan.

### 10. `multimodal_fused_step`
- **Description**: Top-level entry point for fused Vision-Logic multimodal inference.

## `nca::simd` (Infrastructure)

### 11. `Dispatcher<FuncPtr>`
- **Description**: Atomic Jump Cache. Binds function pointers on first use to eliminate CPUID branches in hot loops.

### 12. `CachePolicy<WorkingSetBytes>`
- **Description**: Constexpr compile-time decision tree for tiling and prefetching.
