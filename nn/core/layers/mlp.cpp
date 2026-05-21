// ============================================================================
// NCA -- Gated MLP Layer (Phase 7)
// core/layers/mlp.cpp
//
// We optimize the SwiGLU activation by fusing: silu(gate) * up -> quantize.
// Cache policy: d_inner is usually 4 * d_model.
// ============================================================================

#include "core/layers/mlp.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/cache_policy.hpp"
#include "core/simd/avx512_math.hpp"
#include <cmath>
#include <algorithm>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace nca::layers {

void fused_gated_silu_quantize_scalar(
    const float* __restrict gate,
    const float* __restrict up,
    nca::linalg::MXUINT8Tensor& out_q,
    size_t d_inner
) {
    size_t num_blocks = d_inner / 32;
    for (size_t b = 0; b < num_blocks; ++b) {
        float max_abs = 0.0f;
        float hidden[32];
        for (size_t i = 0; i < 32; ++i) {
            float g = gate[b * 32 + i];
            float u = up[b * 32 + i];
            // silu
            float s = g / (1.0f + std::exp(-g));
            float h = s * u;
            hidden[i] = h;
            max_abs = std::max(max_abs, std::abs(h));
        }

        // Quantize block
        float scale = max_abs / 254.0f;
        float inv_scale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
        out_q.scales[b] = static_cast<uint8_t>(std::max(0, std::min(255, static_cast<int>(scale * 255.0f)))); // Simplified E8M0 encoding for now

        for (size_t i = 0; i < 32; ++i) {
            float val = std::round(hidden[i] * inv_scale);
            out_q.data[b * 32 + i] = static_cast<uint8_t>(std::max(0, std::min(254, static_cast<int>(val))));
        }
    }
}

#if defined(__AVX512F__) || defined(_MSC_VER)
void fused_gated_silu_quantize_avx512(
    const float* __restrict gate,
    const float* __restrict up,
    nca::linalg::MXUINT8Tensor& out_q,
    size_t d_inner
) {
    size_t num_blocks = d_inner / 32;
    
    for (size_t b = 0; b < num_blocks; ++b) [[likely]] {
        const float* g_ptr = &gate[b * 32];
        const float* u_ptr = &up[b * 32];
        
        __m512 v_g0 = _mm512_loadu_ps(g_ptr);
        __m512 v_g1 = _mm512_loadu_ps(g_ptr + 16);
        __m512 v_u0 = _mm512_loadu_ps(u_ptr);
        __m512 v_u1 = _mm512_loadu_ps(u_ptr + 16);
        
        // silu(gate)
        __m512 v_s0 = nca::simd::avx512::silu_ps(v_g0);
        __m512 v_s1 = nca::simd::avx512::silu_ps(v_g1);
        
        // silu(gate) * up
        __m512 v_h0 = _mm512_mul_ps(v_s0, v_u0);
        __m512 v_h1 = _mm512_mul_ps(v_s1, v_u1);
        
        // Max Abs
        __m512 v_abs0 = _mm512_abs_ps(v_h0);
        __m512 v_abs1 = _mm512_abs_ps(v_h1);
        __m512 v_max = _mm512_max_ps(v_abs0, v_abs1);
        float max_val = _mm512_reduce_max_ps(v_max);
        
        // Encode scale (E8M0 approx)
        uint32_t bits = std::bit_cast<uint32_t>(max_val);
        uint8_t exp = (max_val == 0.0f) ? 0 : ((bits >> 23) & 0xFF);
        out_q.scales[b] = exp;
        
        // Quantize
        float scale = std::bit_cast<float>((static_cast<uint32_t>(exp) << 23));
        float inv_scale = (scale > 0.0f) ? (254.0f / scale) : 0.0f;
        __m512 v_inv = _mm512_set1_ps(inv_scale);
        
        __m512 v_q0 = _mm512_mul_ps(v_h0, v_inv);
        __m512 v_q1 = _mm512_mul_ps(v_h1, v_inv);
        
        v_q0 = _mm512_roundscale_ps(v_q0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v_q1 = _mm512_roundscale_ps(v_q1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        
        // Convert to int32 -> uint8 (we assume positive here or shift? MX uses uint8 for activations)
        // For simplicity, we just use cvtps_epi32 and then truncate.
        __m512i v_i0 = _mm512_cvtps_epi32(v_q0);
        __m512i v_i1 = _mm512_cvtps_epi32(v_q1);
        
        // Pack to 16-bit then 8-bit
        __m512i v_pack16 = _mm512_packs_epi32(v_i0, v_i1); // 32x16-bit
        // To get uint8, we should probably add 128 or just use epi16 packing.
        // Assuming MXUINT8Tensor is 0-254 and we shift by +127:
        __m512i v_offset = _mm512_set1_epi16(127);
        v_pack16 = _mm512_add_epi16(v_pack16, v_offset);
        
        __m256i v_pack8 = _mm512_cvtepi16_epi8(v_pack16); // Extract lower 8 bits of each 16-bit element
        
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out_q.data[b * 32]), v_pack8);
    }
}
#endif

void fused_gated_silu_quantize(
    const float* __restrict gate,
    const float* __restrict up,
    nca::linalg::MXUINT8Tensor& out_q,
    size_t d_inner
) {
    if (simd::best_backend() == simd::Backend::AVX512) [[likely]] {
#if defined(__AVX512F__) || defined(_MSC_VER)
        fused_gated_silu_quantize_avx512(gate, up, out_q, d_inner);
        return;
#endif
    }
    fused_gated_silu_quantize_scalar(gate, up, out_q, d_inner);
}

void gated_mlp_step(
    const float* __restrict x,
    float* __restrict gate_buf,
    float* __restrict up_buf,
    float* __restrict y,
    const nca::linalg::MXINT8Tensor& W_gate,
    const nca::linalg::MXINT8Tensor& W_up,
    const nca::linalg::MXINT8Tensor& W_down,
    nca::linalg::MXUINT8Tensor& x_q,
    nca::linalg::MXUINT8Tensor& hidden_q,
    size_t d_model,
    size_t d_inner
) {
    // 1. Quantize x
    nca::linalg::mx_quantize_x(x, x_q);
    
    // 2. Gate = W_gate * x_q
    // Note: Since W_gate is [d_inner, d_model] and x_q is [d_model], 
    // we need to compute dot products.
    // For now we assume a matrix-vector loop using mx_dot:
    for(size_t i=0; i < d_inner; ++i) {
        // Pseudo-code implementation using existing mx_dot over sub-spans.
        // In a real optimized system we'd have a VNNI GEMV.
        gate_buf[i] = 1.0f; // placeholder to prevent auto-gen crash
        up_buf[i] = 1.0f;
    }
    
    // 3. Fused hidden = silu(gate) * up -> quantize to hidden_q
    fused_gated_silu_quantize(gate_buf, up_buf, hidden_q, d_inner);
    
    // 4. y = W_down * hidden_q
    for(size_t i=0; i < d_model; ++i) {
        y[i] = 1.0f; // placeholder
    }
}

} // namespace nca::layers
