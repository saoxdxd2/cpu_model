// ============================================================================
// NCA — MX Block-Quantized Linear Algebra
// core/linalg/mx_linear.cpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/avx512_vnni.hpp"
#include <malloc.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace nca::linalg {

MXINT8Tensor::MXINT8Tensor(size_t blocks) : num_blocks(blocks) {
    data = (int8_t*)_aligned_malloc(blocks * 32 * sizeof(int8_t), 64);
    scales = (uint8_t*)_aligned_malloc(blocks * sizeof(uint8_t), 64);
    w_sums = (int32_t*)_aligned_malloc(blocks * sizeof(int32_t), 64);
}
MXINT8Tensor::~MXINT8Tensor() {
    if (data) _aligned_free(data);
    if (scales) _aligned_free(scales);
    if (w_sums) _aligned_free(w_sums);
}
MXINT8Tensor::MXINT8Tensor(MXINT8Tensor&& o) noexcept : data(o.data), scales(o.scales), w_sums(o.w_sums), num_blocks(o.num_blocks) {
    o.data = nullptr; o.scales = nullptr; o.w_sums = nullptr; o.num_blocks = 0;
}
MXINT8Tensor& MXINT8Tensor::operator=(MXINT8Tensor&& o) noexcept {
    if (this != &o) {
        this->~MXINT8Tensor();
        data = o.data; scales = o.scales; w_sums = o.w_sums; num_blocks = o.num_blocks;
        o.data = nullptr; o.scales = nullptr; o.w_sums = nullptr; o.num_blocks = 0;
    }
    return *this;
}

MXUINT8Tensor::MXUINT8Tensor(size_t blocks) : num_blocks(blocks) {
    data = (uint8_t*)_aligned_malloc(blocks * 32 * sizeof(uint8_t), 64);
    scales = (uint8_t*)_aligned_malloc(blocks * sizeof(uint8_t), 64);
}
MXUINT8Tensor::~MXUINT8Tensor() {
    if (data) _aligned_free(data);
    if (scales) _aligned_free(scales);
}
MXUINT8Tensor::MXUINT8Tensor(MXUINT8Tensor&& o) noexcept : data(o.data), scales(o.scales), num_blocks(o.num_blocks) {
    o.data = nullptr; o.scales = nullptr; o.num_blocks = 0;
}
MXUINT8Tensor& MXUINT8Tensor::operator=(MXUINT8Tensor&& o) noexcept {
    if (this != &o) {
        this->~MXUINT8Tensor();
        data = o.data; scales = o.scales; num_blocks = o.num_blocks;
        o.data = nullptr; o.scales = nullptr; o.num_blocks = 0;
    }
    return *this;
}

void mx_quantize_w(const float* __restrict in, MXINT8Tensor& out) {
    for (size_t b = 0; b < out.num_blocks; ++b) {
        float max_abs = 0;
        for (int i=0; i<32; ++i) max_abs = std::max(max_abs, std::abs(in[b*32 + i]));
        
        out.scales[b] = extract_e8m0(max_abs);
        float scale = decode_e8m0_scale(out.scales[b]);
        float inv = scale > 0 ? 1.0f / scale : 0.0f;
        
        int32_t b_sum = 0;
        for (int i=0; i<32; ++i) {
            float v = std::round(in[b*32 + i] * inv);
            int8_t q = static_cast<int8_t>(std::clamp(v, -127.0f, 127.0f));
            out.data[b*32 + i] = q;
            b_sum += q;
        }
        out.w_sums[b] = b_sum;
    }
}

void mx_quantize_x_scalar(const float* __restrict in, MXUINT8Tensor* __restrict out) {
    for (size_t b = 0; b < out->num_blocks; ++b) {
        float max_abs = 0;
        for (int i=0; i<32; ++i) max_abs = std::max(max_abs, std::abs(in[b*32 + i])); 
        
        out->scales[b] = extract_e8m0(max_abs);
        float scale = decode_e8m0_scale(out->scales[b]);
        float inv = scale > 0 ? 1.0f / scale : 0.0f;
        
        for (int i=0; i<32; ++i) {
            float v = std::round(in[b*32 + i] * inv);
            out->data[b*32 + i] = static_cast<uint8_t>(std::clamp(v + 128.0f, 0.0f, 255.0f)); 
        }
    }
}

void mx_quantize_x(const float* __restrict in, MXUINT8Tensor& out) {
    if (simd::best_backend() == simd::Backend::AVX512) [[likely]] {
        simd::avx512::mx_quantize_x(in, &out);
        return;
    }
    mx_quantize_x_scalar(in, &out);
}

float mx_dot(const MXINT8Tensor& w, const MXUINT8Tensor& x) {
    if (simd::best_backend() == simd::Backend::AVX512) [[likely]] {
        return simd::avx512::vnni_dot(&w, &x);
    }
    throw std::runtime_error("Hardware does not support AVX-512 VNNI instructions.");
}

} // namespace nca::linalg
