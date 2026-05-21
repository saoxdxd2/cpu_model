// ============================================================================
// NCA -- Vision Stage 1: Global Scanner (Phase 8)
// core/vision/scanner.cpp
// ============================================================================

#include "core/vision/scanner.hpp"
#include "core/layers/ssm.hpp"
#include "core/simd/dispatch.hpp"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace nca::vision {

void dwconv2d_3x3_scalar(
    const float* __restrict input,
    const float* __restrict weight,
    float* __restrict output,
    ScannerConfig cfg
) {
    auto H = cfg.H;
    auto W = cfg.W;
    auto C = cfg.C;

    for (size_t y = 1; y < H - 1; ++y) {
        for (size_t x = 1; x < W - 1; ++x) {
            for (size_t c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        sum += input[((y + ky) * W + (x + kx)) * C + c] * weight[((ky+1) * 3 + (kx+1)) * C + c];
                    }
                }
                output[(y * W + x) * C + c] = sum;
            }
        }
    }
}

#if defined(__AVX512F__) || defined(_MSC_VER)
void dwconv2d_3x3_avx512(
    const float* __restrict input,
    const float* __restrict weight,
    float* __restrict output,
    ScannerConfig cfg
) {
    auto H = cfg.H;
    auto W = cfg.W;
    auto C = cfg.C;

    for (size_t y = 1; y < H - 1; ++y) [[likely]] {
        size_t x = 1;
        // 2x Spatial Unrolling for the X dimension
        for (; x + 1 < W - 1; x += 2) {
            for (size_t c = 0; c + 15 < C; c += 16) {
                auto sum0 = _mm512_setzero_ps();
                auto sum1 = _mm512_setzero_ps();
                
                auto apply_kernel = [&](auto& s, size_t ox) {
                    for (int ky = -1; ky <= 1; ++ky) {
                        for (int kx = -1; kx <= 1; ++kx) {
                            s = _mm512_fmadd_ps(_mm512_loadu_ps(&input[((y+ky) * W + (ox+kx)) * C + c]), 
                                               _mm512_loadu_ps(&weight[((ky+1) * 3 + (kx+1)) * C + c]), s);
                        }
                    }
                };

                apply_kernel(sum0, x);
                apply_kernel(sum1, x + 1);
                
                _mm512_storeu_ps(&output[(y * W + x) * C + c], sum0);
                _mm512_storeu_ps(&output[(y * W + (x+1)) * C + c], sum1);
            }
        }
        for (; x < W - 1; ++x) {
            for (size_t c = 0; c + 15 < C; c += 16) {
                auto sum = _mm512_setzero_ps();
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        sum = _mm512_fmadd_ps(_mm512_loadu_ps(&input[((y+ky) * W + (x+kx)) * C + c]), 
                                           _mm512_loadu_ps(&weight[((ky+1) * 3 + (kx+1)) * C + c]), sum);
                    }
                }
                _mm512_storeu_ps(&output[(y * W + x) * C + c], sum);
            }
        }
    }
}
#endif

void dwconv2d_3x3(
    const float* __restrict input,
    const float* __restrict weight,
    float* __restrict output,
    ScannerConfig cfg
) {
    if (cfg.C % 16 == 0 && simd::best_backend() == simd::Backend::AVX512) [[likely]] {
#if defined(__AVX512F__) || defined(_MSC_VER)
        dwconv2d_3x3_avx512(input, weight, output, cfg);
        return;
#endif
    }
    dwconv2d_3x3_scalar(input, weight, output, cfg);
}

void ssm2d_scan(
    float* __restrict h,
    const float* __restrict A,
    const float* __restrict B,
    const float* __restrict C_proj,
    const float* __restrict x,
    float* __restrict y,
    ScannerConfig cfg
) {
    nca::layers::SSMConfig ssm_cfg;
    ssm_cfg.d_inner = cfg.H * cfg.W * cfg.C;
    ssm_cfg.d_state = 16;
    nca::layers::ssm_step(h, A, B, C_proj, x, y, ssm_cfg);
}

} // namespace nca::vision
