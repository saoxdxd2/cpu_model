#pragma once
// ============================================================================
// NCA — MX Block-Quantized Linear Algebra
// core/linalg/mx_linear.hpp
// ============================================================================

#include "core/simd/memory.hpp"
#include <cstdint>
#include <cstddef>
#include <bit>
#include <memory>
#include <algorithm>

namespace nca::linalg {

inline float decode_e8m0_scale(uint8_t e) {
    return (e == 0) ? 0.0f : std::bit_cast<float>(static_cast<uint32_t>(e) << 23);
}

inline uint8_t extract_e8m0(float max_abs) {
    if (max_abs == 0.0f) return 0;
    auto bits = std::bit_cast<uint32_t>(max_abs / 127.0f);
    return static_cast<uint8_t>((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) > 0);
}

// ── RAII MX Tensors ──────────────────────────────────────────────────────────
template <typename T>
struct MXTensorBase {
    T* data = nullptr;
    uint8_t* scales = nullptr;
    size_t num_blocks = 0;
    nca::simd::aligned_unique_ptr<T[]> data_owner;
    nca::simd::aligned_unique_ptr<uint8_t[]> scales_owner;

    MXTensorBase() = default;
    MXTensorBase(size_t b, size_t n) : num_blocks(b) {
        if (b > 0) {
            data_owner = nca::simd::make_aligned_unique<T[]>(b * n);
            scales_owner = nca::simd::make_aligned_unique<uint8_t[]>(b);
            data = data_owner.get(); scales = scales_owner.get();
        }
    }
};

struct MXINT8Tensor : public MXTensorBase<int8_t> {
    int32_t* w_sums = nullptr;
    nca::simd::aligned_unique_ptr<int32_t[]> w_sums_owner;

    MXINT8Tensor() = default;
    explicit MXINT8Tensor(size_t b) : MXTensorBase<int8_t>(b, 32) {
        if (b > 0) {
            w_sums_owner = nca::simd::make_aligned_unique<int32_t[]>(b);
            w_sums = w_sums_owner.get();
        }
    }
};

struct MXUINT8Tensor : public MXTensorBase<uint8_t> {
    MXUINT8Tensor() = default;
    explicit MXUINT8Tensor(size_t b) : MXTensorBase<uint8_t>(b, 32) {}
};

// ── Kernel API ───────────────────────────────────────────────────────────────
void mx_quantize_w(const float* __restrict in, MXINT8Tensor& out);
void mx_quantize_x(const float* __restrict in, MXUINT8Tensor& out);
void mx_fused_silu_quantize_x(const float* __restrict in, MXUINT8Tensor& out);

// Local Gaussian Optimizer: delta_W = lr * error * x / (var(x) + eps)
void mx_update_gaussian_moment(
    MXINT8Tensor& w, 
    const MXUINT8Tensor& x_q, 
    float error, 
    float lr,
    float precomputed_norm = 0.0f // 0 means compute internally
);

// Computes the Var(x) proxy (Power) once for multiple expert updates
float mx_compute_activation_norm(const MXUINT8Tensor& x_q);

float mx_dot(const MXINT8Tensor& w, const MXUINT8Tensor& x);
void  mx_dual_dot(const MXINT8Tensor& w0, const MXINT8Tensor& w1, const MXUINT8Tensor& x, float& o0, float& o1);
void  mx_quad_dot(const MXINT8Tensor& w0, const MXINT8Tensor& w1, const MXINT8Tensor& w2, const MXINT8Tensor& w3, const MXUINT8Tensor& x, float& o0, float& o1, float& o2, float& o3);

void mx_rank16_dot(
    const MXINT8Tensor* __restrict weights, 
    const MXUINT8Tensor& x,
    float* __restrict outputs               
);

void mx_rank16_dot_ptrs(
    const MXINT8Tensor** __restrict weights, 
    const MXUINT8Tensor& x,
    float* __restrict outputs               
);

void  mx_gemv(const MXINT8Tensor& W, const MXUINT8Tensor& x, float* y, size_t rows, size_t cols);

} // namespace nca::linalg
