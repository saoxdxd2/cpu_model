// ============================================================================
// NCA — MX Block-Quantized Linear Algebra
// core/linalg/mx_linear.cpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/avx512_vnni.hpp"
#include <malloc.h>
#include <cmath>
#include <bit>
#include <algorithm>
#include <iostream>

namespace nca::linalg {

// E8M0 Math
inline uint8_t extract_e8m0(float max_abs) {
    if (max_abs == 0.0f) return 0;
    float scale = max_abs / 127.0f;
    uint32_t bits = std::bit_cast<uint32_t>(scale);
    uint32_t mantissa = bits & 0x7FFFFF;
    uint8_t exp = (bits >> 23) & 0xFF;
    if (mantissa > 0) exp += 1; // Round up
    return exp;
}

inline float decode_scale(uint8_t e) {
    if (e == 0) return 0.0f;
    uint32_t bits = static_cast<uint32_t>(e) << 23;
    return std::bit_cast<float>(bits);
}

void mx_alloc_int8(MXINT8Tensor* t, size_t num_blocks) {
    t->num_blocks = num_blocks;
    t->data = (int8_t*)_aligned_malloc(num_blocks * 32 * sizeof(int8_t), 64);
    t->scales = (uint8_t*)_aligned_malloc(num_blocks * sizeof(uint8_t), 64);
}

void mx_free_int8(MXINT8Tensor* t) {
    _aligned_free(t->data);
    _aligned_free(t->scales);
    t->data = nullptr;
    t->scales = nullptr;
}

void mx_alloc_uint8(MXUINT8Tensor* t, size_t num_blocks) {
    t->num_blocks = num_blocks;
    t->data = (uint8_t*)_aligned_malloc(num_blocks * 32 * sizeof(uint8_t), 64);
    t->scales = (uint8_t*)_aligned_malloc(num_blocks * sizeof(uint8_t), 64);
}

void mx_free_uint8(MXUINT8Tensor* t) {
    _aligned_free(t->data);
    _aligned_free(t->scales);
    t->data = nullptr;
    t->scales = nullptr;
}

void mx_quantize_w(const float* __restrict in, MXINT8Tensor* __restrict out) {
    for (size_t b = 0; b < out->num_blocks; ++b) {
        float max_abs = 0;
        for (int i=0; i<32; ++i) max_abs = std::max(max_abs, std::abs(in[b*32 + i]));
        
        out->scales[b] = extract_e8m0(max_abs);
        float scale = decode_scale(out->scales[b]);
        float inv = scale > 0 ? 1.0f / scale : 0.0f;
        
        for (int i=0; i<32; ++i) {
            float v = std::round(in[b*32 + i] * inv);
            out->data[b*32 + i] = static_cast<int8_t>(std::clamp(v, -127.0f, 127.0f));
        }
    }
}

void mx_quantize_x(const float* __restrict in, MXUINT8Tensor* __restrict out) {
    for (size_t b = 0; b < out->num_blocks; ++b) {
        float max_abs = 0;
        for (int i=0; i<32; ++i) max_abs = std::max(max_abs, std::abs(in[b*32 + i])); 
        
        out->scales[b] = extract_e8m0(max_abs);
        float scale = decode_scale(out->scales[b]);
        float inv = scale > 0 ? 1.0f / scale : 0.0f;
        
        for (int i=0; i<32; ++i) {
            float v = std::round(std::abs(in[b*32 + i]) * inv); // Assume positive or abs for now
            out->data[b*32 + i] = static_cast<uint8_t>(std::clamp(v, 0.0f, 127.0f)); 
        }
    }
}

float mx_dot(const MXINT8Tensor* __restrict w, const MXUINT8Tensor* __restrict x) {
    using simd::Backend;
    if (simd::best_backend() == Backend::AVX512) [[likely]] {
        return simd::avx512::vnni_dot(w, x);
    }
    
    std::cerr << "CRITICAL ERROR: AVX-512 VNNI is required for MXINT8 operations. Scalar fallback missing.\n";
    return 0.0f; 
}

} // namespace nca::linalg
