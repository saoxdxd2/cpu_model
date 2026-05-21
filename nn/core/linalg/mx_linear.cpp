// ============================================================================
// NCA — MX Block-Quantized Linear Algebra
// core/linalg/mx_linear.cpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/avx512_vnni.hpp"
#include "core/log.hpp"
#include <malloc.h>
#include <cmath>
#include <algorithm>

namespace nca::linalg {

// ── RAII Tensor Lifecycle ────────────────────────────────────────────────────

MXINT8Tensor::MXINT8Tensor(size_t blocks) : num_blocks(blocks) {
    void* d = _aligned_malloc(blocks * 32 * sizeof(int8_t),  64);
    void* s = _aligned_malloc(blocks * sizeof(uint8_t),      64);
    void* w = _aligned_malloc(blocks * sizeof(int32_t),      64);
    if (!d || !s || !w) throw std::bad_alloc();
    
    data_owner.reset((int8_t*)d);
    scales_owner.reset((uint8_t*)s);
    w_sums_owner.reset((int32_t*)w);
    
    data = data_owner.get();
    scales = scales_owner.get();
    w_sums = w_sums_owner.get();
}

MXINT8Tensor::MXINT8Tensor(MXINT8Tensor&& o) noexcept
    : data(o.data), scales(o.scales), w_sums(o.w_sums), num_blocks(o.num_blocks),
      data_owner(std::move(o.data_owner)),
      scales_owner(std::move(o.scales_owner)),
      w_sums_owner(std::move(o.w_sums_owner)) {
    o.data = nullptr; o.scales = nullptr; o.w_sums = nullptr; o.num_blocks = 0;
}

MXINT8Tensor& MXINT8Tensor::operator=(MXINT8Tensor&& o) noexcept {
    if (this != &o) {
        data = o.data; scales = o.scales; w_sums = o.w_sums; num_blocks = o.num_blocks;
        data_owner = std::move(o.data_owner);
        scales_owner = std::move(o.scales_owner);
        w_sums_owner = std::move(o.w_sums_owner);
        o.data = nullptr; o.scales = nullptr; o.w_sums = nullptr; o.num_blocks = 0;
    }
    return *this;
}

MXUINT8Tensor::MXUINT8Tensor(size_t blocks) : num_blocks(blocks) {
    void* d = _aligned_malloc(blocks * 32 * sizeof(uint8_t), 64);
    void* s = _aligned_malloc(blocks * sizeof(uint8_t),      64);
    if (!d || !s) throw std::bad_alloc();
    
    data_owner.reset((uint8_t*)d);
    scales_owner.reset((uint8_t*)s);
    
    data = data_owner.get();
    scales = scales_owner.get();
}

MXUINT8Tensor::MXUINT8Tensor(MXUINT8Tensor&& o) noexcept
    : data(o.data), scales(o.scales), num_blocks(o.num_blocks),
      data_owner(std::move(o.data_owner)),
      scales_owner(std::move(o.scales_owner)) {
    o.data = nullptr; o.scales = nullptr; o.num_blocks = 0;
}

MXUINT8Tensor& MXUINT8Tensor::operator=(MXUINT8Tensor&& o) noexcept {
    if (this != &o) {
        data = o.data; scales = o.scales; num_blocks = o.num_blocks;
        data_owner = std::move(o.data_owner);
        scales_owner = std::move(o.scales_owner);
        o.data = nullptr; o.scales = nullptr; o.num_blocks = 0;
    }
    return *this;
}

// ── Weight Quantization ─────────────────────────────────────────────────────

void mx_quantize_w(const float* __restrict in, MXINT8Tensor& out) {
    for (size_t b = 0; b < out.num_blocks; ++b) {
        float max_abs = 0;
        for (int i = 0; i < 32; ++i)
            max_abs = std::max(max_abs, std::abs(in[b * 32 + i]));

        out.scales[b] = extract_e8m0(max_abs);
        auto scale = decode_e8m0_scale(out.scales[b]);
        auto inv = scale > 0 ? 1.0f / scale : 0.0f;

        int32_t b_sum = 0;
        for (int i = 0; i < 32; ++i) {
            auto v = std::round(in[b * 32 + i] * inv);
            auto q = static_cast<int8_t>(std::clamp(v, -127.0f, 127.0f));
            out.data[b * 32 + i] = q;
            b_sum += q;
        }
        out.w_sums[b] = b_sum;
    }
}

void mx_quantize_x_scalar(const float* __restrict in, MXUINT8Tensor* __restrict out) {
    for (size_t b = 0; b < out->num_blocks; ++b) {
        float max_abs = 0;
        for (int i = 0; i < 32; ++i)
            max_abs = std::max(max_abs, std::abs(in[b * 32 + i]));

        out->scales[b] = extract_e8m0(max_abs);
        auto scale = decode_e8m0_scale(out->scales[b]);
        auto inv = scale > 0 ? 1.0f / scale : 0.0f;

        for (int i = 0; i < 32; ++i) {
            auto v = std::round(in[b * 32 + i] * inv);
            out->data[b * 32 + i] = static_cast<uint8_t>(std::clamp(v + 128.0f, 0.0f, 255.0f));
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

void mx_fused_silu_quantize_x_scalar(const float* __restrict in, MXUINT8Tensor* __restrict out) {
    for (size_t b = 0; b < out->num_blocks; ++b) {
        float max_abs = 0;
        float silu_cache[32];
        for (int i = 0; i < 32; ++i) {
            auto x = in[b * 32 + i];
            auto s = x / (1.0f + std::exp(-x));
            silu_cache[i] = s;
            max_abs = std::max(max_abs, std::abs(s));
        }

        out->scales[b] = extract_e8m0(max_abs);
        auto scale = decode_e8m0_scale(out->scales[b]);
        auto inv = scale > 0 ? 1.0f / scale : 0.0f;

        for (int i = 0; i < 32; ++i) {
            auto v = std::round(silu_cache[i] * inv);
            out->data[b * 32 + i] = static_cast<uint8_t>(std::clamp(v + 128.0f, 0.0f, 255.0f));
        }
    }
}

void mx_fused_silu_quantize_x(const float* __restrict in, MXUINT8Tensor& out) {
    if (simd::best_backend() == simd::Backend::AVX512) [[likely]] {
        simd::avx512::mx_fused_silu_quantize_x(in, &out);
        return;
    }
    mx_fused_silu_quantize_x_scalar(in, &out);
}

static float mx_dot_scalar(const MXINT8Tensor& w, const MXUINT8Tensor& x) {
    float sum = 0.0f;
    auto num_blocks = std::min(w.num_blocks, x.num_blocks);
    for (size_t b = 0; b < num_blocks; ++b) {
        float block_sum = 0.0f;
        for (int i = 0; i < 32; ++i) {
            int val_x = static_cast<int>(x.data[b * 32 + i]) - 128;
            int val_w = static_cast<int>(w.data[b * 32 + i]);
            block_sum += static_cast<float>(val_x * val_w);
        }
        sum += block_sum * decode_e8m0_scale(w.scales[b]) * decode_e8m0_scale(x.scales[b]);
    }
    return sum;
}

float mx_dot(const MXINT8Tensor& w, const MXUINT8Tensor& x) {
    if (simd::best_backend() == simd::Backend::AVX512) [[likely]] {
        return simd::avx512::vnni_dot(&w, &x);
    }
    return mx_dot_scalar(w, x);
}

void mx_dual_dot(
    const MXINT8Tensor& w0,
    const MXINT8Tensor& w1,
    const MXUINT8Tensor& x,
    float& out0,
    float& out1
) {
    if (simd::best_backend() == simd::Backend::AVX512) [[likely]] {
        simd::avx512::dual_vnni_dot(&w0, &w1, &x, out0, out1);
        return;
    }
    out0 = mx_dot_scalar(w0, x);
    out1 = mx_dot_scalar(w1, x);
}

void mx_quad_dot(
    const MXINT8Tensor& w0,
    const MXINT8Tensor& w1,
    const MXINT8Tensor& w2,
    const MXINT8Tensor& w3,
    const MXUINT8Tensor& x,
    float& out0,
    float& out1,
    float& out2,
    float& out3
) {
    if (simd::best_backend() == simd::Backend::AVX512) [[likely]] {
        simd::avx512::quad_vnni_dot(&w0, &w1, &w2, &w3, &x, out0, out1, out2, out3);
        return;
    }
    out0 = mx_dot_scalar(w0, x);
    out1 = mx_dot_scalar(w1, x);
    out2 = mx_dot_scalar(w2, x);
    out3 = mx_dot_scalar(w3, x);
}

void mx_gemv(const MXINT8Tensor& W, const MXUINT8Tensor& x, float* y, size_t rows, size_t cols) {
    const size_t blocks_per_row = cols / 32;

    if (simd::best_backend() == simd::Backend::AVX512) [[likely]] {
        // ── QUAD-ROW UNROLLING (Compute Tweak) ───────────────────────────────
        // We process 4 rows at once to reuse 'x' blocks 4x in L1 registers.
        size_t r = 0;
        for (; r + 3 < rows; r += 4) [[likely]] {
            MXINT8Tensor v0, v1, v2, v3;
            auto prep = [&](MXINT8Tensor& v, size_t ri) {
                v.data = const_cast<int8_t*>(W.data + ri * cols);
                v.scales = const_cast<uint8_t*>(W.scales + ri * blocks_per_row);
                v.w_sums = const_cast<int32_t*>(W.w_sums + ri * blocks_per_row);
                v.num_blocks = blocks_per_row;
            };
            prep(v0, r); prep(v1, r + 1); prep(v2, r + 2); prep(v3, r + 3);
            
            mx_quad_dot(v0, v1, v2, v3, x, y[r], y[r+1], y[r+2], y[r+3]);

            v0.data = v1.data = v2.data = v3.data = nullptr;
        }
        // Dual tail
        for (; r + 1 < rows; r += 2) {
            MXINT8Tensor v0, v1;
            auto prep = [&](MXINT8Tensor& v, size_t ri) {
                v.data = const_cast<int8_t*>(W.data + ri * cols);
                v.scales = const_cast<uint8_t*>(W.scales + ri * blocks_per_row);
                v.w_sums = const_cast<int32_t*>(W.w_sums + ri * blocks_per_row);
                v.num_blocks = blocks_per_row;
            };
            prep(v0, r); prep(v1, r + 1);
            mx_dual_dot(v0, v1, x, y[r], y[r+1]);
            v0.data = v1.data = nullptr;
        }
        // Single tail
        for (; r < rows; ++r) {
            MXINT8Tensor v;
            v.data = const_cast<int8_t*>(W.data + r * cols);
            v.scales = const_cast<uint8_t*>(W.scales + r * blocks_per_row);
            v.w_sums = const_cast<int32_t*>(W.w_sums + r * blocks_per_row);
            v.num_blocks = blocks_per_row;
            y[r] = mx_dot(v, x);
            v.data = nullptr;
        }
    } else {

        for (size_t r = 0; r < rows; ++r) {
            float sum = 0.0f;
            for (size_t b = 0; b < blocks_per_row; ++b) {
                auto wb = r * blocks_per_row + b;
                float block_sum = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    int val_x = static_cast<int>(x.data[b * 32 + i]) - 128;
                    int val_w = static_cast<int>(W.data[wb * 32 + i]);
                    block_sum += static_cast<float>(val_x * val_w);
                }
                sum += block_sum * decode_e8m0_scale(W.scales[wb]) * decode_e8m0_scale(x.scales[b]);
            }
            y[r] = sum;
        }
    }
}

} // namespace nca::linalg
