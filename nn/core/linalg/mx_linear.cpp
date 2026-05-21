// ============================================================================
// NCA — MX Block-Quantized Linear Algebra
// core/linalg/mx_linear.cpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/avx512_vnni.hpp"
#include "core/log.hpp"
#include <cmath>
#include <algorithm>

namespace nca::linalg {

// ── RAII Tensor Lifecycle ────────────────────────────────────────────────────

MXINT8Tensor::MXINT8Tensor(size_t blocks) : num_blocks(blocks) {
    data_ptr   = nca::simd::make_aligned_unique<int8_t>(blocks * 32);
    scales_ptr = nca::simd::make_aligned_unique<uint8_t>(blocks);
    w_sums_ptr = nca::simd::make_aligned_unique<int32_t>(blocks);
    
    data   = data_ptr.get();
    scales = scales_ptr.get();
    w_sums = w_sums_ptr.get();
}

MXINT8Tensor::MXINT8Tensor(MXINT8Tensor&& o) noexcept
    : data(o.data), scales(o.scales), w_sums(o.w_sums), num_blocks(o.num_blocks),
      data_ptr(std::move(o.data_ptr)),
      scales_ptr(std::move(o.scales_ptr)),
      w_sums_ptr(std::move(o.w_sums_ptr)) {
    o.data = nullptr; o.scales = nullptr; o.w_sums = nullptr; o.num_blocks = 0;
}

MXINT8Tensor& MXINT8Tensor::operator=(MXINT8Tensor&& o) noexcept {
    if (this != &o) {
        data = o.data; scales = o.scales; w_sums = o.w_sums; num_blocks = o.num_blocks;
        data_ptr = std::move(o.data_ptr);
        scales_ptr = std::move(o.scales_ptr);
        w_sums_ptr = std::move(o.w_sums_ptr);
        o.data = nullptr; o.scales = nullptr; o.w_sums = nullptr; o.num_blocks = 0;
    }
    return *this;
}

MXUINT8Tensor::MXUINT8Tensor(size_t blocks) : num_blocks(blocks) {
    data_ptr   = nca::simd::make_aligned_unique<uint8_t>(blocks * 32);
    scales_ptr = nca::simd::make_aligned_unique<uint8_t>(blocks);
    
    data   = data_ptr.get();
    scales = scales_ptr.get();
}

MXUINT8Tensor::MXUINT8Tensor(MXUINT8Tensor&& o) noexcept
    : data(o.data), scales(o.scales), num_blocks(o.num_blocks),
      data_ptr(std::move(o.data_ptr)),
      scales_ptr(std::move(o.scales_ptr)) {
    o.data = nullptr; o.scales = nullptr; o.num_blocks = 0;
}

MXUINT8Tensor& MXUINT8Tensor::operator=(MXUINT8Tensor&& o) noexcept {
    if (this != &o) {
        data = o.data; scales = o.scales; num_blocks = o.num_blocks;
        data_ptr = std::move(o.data_ptr);
        scales_ptr = std::move(o.scales_ptr);
        o.data = nullptr; o.scales = nullptr; o.num_blocks = 0;
    }
    return *this;
}

// ── Weight Quantization ─────────────────────────────────────────────────────

void mx_quantize_w(const float* __restrict in, MXINT8Tensor& out) {
    for (size_t b = 0; b < out.num_blocks; ++b) {
        float max_abs = 0;
        for (int i = 0; i < 32; i += 4) {
            max_abs = std::max({max_abs, std::abs(in[b * 32 + i]), std::abs(in[b * 32 + i + 1]), std::abs(in[b * 32 + i + 2]), std::abs(in[b * 32 + i + 3])});
        }

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

// ── Activation Quantization ─────────────────────────────────────────────────

void mx_quantize_x_scalar(const float* __restrict in, MXUINT8Tensor* __restrict out) {
    for (size_t b = 0; b < out->num_blocks; ++b) {
        float max_abs = 0;
        for (int i = 0; i < 32; i += 4) {
            max_abs = std::max({max_abs, std::abs(in[b * 32 + i]), std::abs(in[b * 32 + i + 1]), std::abs(in[b * 32 + i + 2]), std::abs(in[b * 32 + i + 3])});
        }

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

// ── Fused SiLU + Quantize ───────────────────────────────────────────────────

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

// ── Dot Product ─────────────────────────────────────────────────────────────

static float mx_dot_scalar(const MXINT8Tensor& w, const MXUINT8Tensor& x) {
    float sum = 0.0f;
    auto num_blocks = std::min(w.num_blocks, x.num_blocks);
    for (size_t b = 0; b < num_blocks; ++b) {
        float block_sum = 0.0f;
        for (int i = 0; i < 32; i += 8) {
            block_sum += static_cast<float>((static_cast<int>(x.data[b * 32 + i]) - 128) * static_cast<int>(w.data[b * 32 + i]));
            block_sum += static_cast<float>((static_cast<int>(x.data[b * 32 + i + 1]) - 128) * static_cast<int>(w.data[b * 32 + i + 1]));
            block_sum += static_cast<float>((static_cast<int>(x.data[b * 32 + i + 2]) - 128) * static_cast<int>(w.data[b * 32 + i + 2]));
            block_sum += static_cast<float>((static_cast<int>(x.data[b * 32 + i + 3]) - 128) * static_cast<int>(w.data[b * 32 + i + 3]));
            block_sum += static_cast<float>((static_cast<int>(x.data[b * 32 + i + 4]) - 128) * static_cast<int>(w.data[b * 32 + i + 4]));
            block_sum += static_cast<float>((static_cast<int>(x.data[b * 32 + i + 5]) - 128) * static_cast<int>(w.data[b * 32 + i + 5]));
            block_sum += static_cast<float>((static_cast<int>(x.data[b * 32 + i + 6]) - 128) * static_cast<int>(w.data[b * 32 + i + 6]));
            block_sum += static_cast<float>((static_cast<int>(x.data[b * 32 + i + 7]) - 128) * static_cast<int>(w.data[b * 32 + i + 7]));
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

// ── GEMV: Zero-Cost View + L1 Tiling ────────────────────────────────────────

void mx_gemv(const MXINT8Tensor& W, const MXUINT8Tensor& x, float* y, size_t rows, size_t cols) {
    const auto blocks_per_row = cols / 32;

    if (simd::best_backend() == simd::Backend::AVX512) [[likely]] {
        constexpr size_t ROW_TILE = 32;
        for (size_t rt = 0; rt < rows; rt += ROW_TILE) {
            auto r_end = std::min(rt + ROW_TILE, rows);
            for (size_t r = rt; r < r_end; ++r) {
                if (r + 1 < rows) {
                    _mm_prefetch((const char*)(W.data + (r + 1) * cols), _MM_HINT_T0);
                    _mm_prefetch((const char*)(W.scales + (r + 1) * blocks_per_row), _MM_HINT_T0);
                }

                MXINT8Tensor view;
                view.data    = const_cast<int8_t*>(W.data + r * cols);
                view.scales  = const_cast<uint8_t*>(W.scales + r * blocks_per_row);
                view.w_sums  = const_cast<int32_t*>(W.w_sums + r * blocks_per_row);
                view.num_blocks = blocks_per_row;

                y[r] = simd::avx512::vnni_dot(&view, &x);

                view.data = nullptr; view.scales = nullptr; view.w_sums = nullptr;
            }
        }
    } else {
        for (size_t r = 0; r < rows; ++r) {
            float sum = 0.0f;
            for (size_t b = 0; b < blocks_per_row; ++b) {
                auto wb = r * blocks_per_row + b;
                float block_sum = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    block_sum += static_cast<float>((static_cast<int>(x.data[b * 32 + i]) - 128) * static_cast<int>(W.data[wb * 32 + i]));
                }
                sum += block_sum * decode_e8m0_scale(W.scales[wb]) * decode_e8m0_scale(x.scales[b]);
            }
            y[r] = sum;
        }
    }
}

} // namespace nca::linalg
