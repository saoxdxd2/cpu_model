#pragma once
// ============================================================================
// NCA — MX Block-Quantized Linear Algebra
// core/linalg/mx_linear.hpp
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <bit>

namespace nca::linalg {

inline float decode_e8m0_scale(uint8_t e) {
    if (e == 0) return 0.0f;
    uint32_t bits = static_cast<uint32_t>(e) << 23;
    return std::bit_cast<float>(bits);
}

inline uint8_t extract_e8m0(float max_abs) {
    if (max_abs == 0.0f) return 0;
    float scale = max_abs / 127.0f;
    uint32_t bits = std::bit_cast<uint32_t>(scale);
    uint32_t mantissa = bits & 0x7FFFFF;
    uint8_t exp = (bits >> 23) & 0xFF;
    if (mantissa > 0) exp += 1;
    return exp;
}

// RAII wrapper for 64-byte aligned tensors to prevent manual lifecycle ceremony
struct MXINT8Tensor {
    int8_t* __restrict data = nullptr;
    uint8_t* __restrict scales = nullptr;
    int32_t* __restrict w_sums = nullptr; // Precomputed sums for activation zero-point shift
    size_t num_blocks = 0;

    MXINT8Tensor() = default;
    explicit MXINT8Tensor(size_t blocks);
    ~MXINT8Tensor();

    MXINT8Tensor(const MXINT8Tensor&) = delete;
    MXINT8Tensor& operator=(const MXINT8Tensor&) = delete;

    MXINT8Tensor(MXINT8Tensor&& o) noexcept;
    MXINT8Tensor& operator=(MXINT8Tensor&& o) noexcept;
};

struct MXUINT8Tensor {
    uint8_t* __restrict data = nullptr;
    uint8_t* __restrict scales = nullptr;
    size_t num_blocks = 0;

    MXUINT8Tensor() = default;
    explicit MXUINT8Tensor(size_t blocks);
    ~MXUINT8Tensor();

    MXUINT8Tensor(const MXUINT8Tensor&) = delete;
    MXUINT8Tensor& operator=(const MXUINT8Tensor&) = delete;

    MXUINT8Tensor(MXUINT8Tensor&& o) noexcept;
    MXUINT8Tensor& operator=(MXUINT8Tensor&& o) noexcept;
};

void mx_quantize_w(const float* __restrict in, MXINT8Tensor& out);
void mx_quantize_x(const float* __restrict in, MXUINT8Tensor& out);
void mx_fused_silu_quantize_x(const float* __restrict in, MXUINT8Tensor& out);

float mx_dot(const MXINT8Tensor& w, const MXUINT8Tensor& x);

// Matrix-Vector Multiplication: y = W * x
// W: [rows, cols] (MXINT8)
// x: [cols] (MXUINT8)
// y: [rows] (float)
void mx_gemv(const MXINT8Tensor& W, const MXUINT8Tensor& x, float* y, size_t rows, size_t cols);

} // namespace nca::linalg
