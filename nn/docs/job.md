# NCA Function Registry & API Contract (job.md)

This document tracks all built and verified mathematical functions within the NCA project.

## `nca::linalg` (The Saturated Math Core)

### 1. `mx_quad_dot`
- **Signature**: `void mx_quad_dot(const MXINT8Tensor& w0, ..., const MXUINT8Tensor& x, float& o0, ...)`
- **Trick**: **Geometric Register Reuse**. Processes 4 rows against 1 shared activation vector.
- **Performance**: **2.3 us** for 1024D (performing 4 dot products). Faster than a single scalar dot!
- **Status**: **SATURATED**. Hits L1-load port limits.

### 2. `mx_gemv`
- **Signature**: `void mx_gemv(const MXINT8Tensor& W, const MXUINT8Tensor& x, float* y, size_t rows, size_t cols)`
- **Performance**: **139 us** for 1024x1024.
- **Tricks**: Rank-4 sequential scan, software prefetching, and log-space scale fusion.

## `nca::layers` (The Fused Backbone)

### 3. `glr_step`
- **Performance**: **0.26 cycles/element**.
- **Status**: **ALU-Bound**. Cannot be optimized further on Ice Lake.

### 4. `halting_step`
- **Performance**: **1.7 us**. 
- **Trick**: Branchless AVX-512 minimax sigmoid proxy.

## `nca::execution` (Routing & Fusion)

### 5. `shuffle_active_tokens`
- **Performance**: **5.1 us**.
- **Trick**: SIMD physical reordering to enable sequential scans.

### 6. `multimodal_fused_step`
- **Description**: Horizontal core fusion of Vision SSM and Logic Quantization.
- **Impact**: Zero memory round-trips for intermediate multimodal activations.
