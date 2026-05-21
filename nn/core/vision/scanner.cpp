// ============================================================================
// NCA -- Vision Stage 1: Global Scanner (Phase 8)
// core/vision/scanner.cpp
//
// Fast NHWC 3x3 Depthwise Convolution and 2D flattened SSM.
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
    size_t H = cfg.H;
    size_t W = cfg.W;
    size_t C = cfg.C;

    for (size_t y = 1; y < H - 1; ++y) {
        for (size_t x = 1; x < W - 1; ++x) {
            for (size_t c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        size_t iy = y + ky;
                        size_t ix = x + kx;
                        sum += input[(iy * W + ix) * C + c] * weight[((ky+1) * 3 + (kx+1)) * C + c];
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
    size_t H = cfg.H;
    size_t W = cfg.W;
    size_t C = cfg.C;

    for (size_t y = 1; y < H - 1; ++y) [[likely]] {
        for (size_t x = 1; x < W - 1; ++x) {
            for (size_t c = 0; c + 15 < C; c += 16) {
                __m512 sum = _mm512_setzero_ps();
                
                // Fully unroll the 3x3 kernel (9 FMAs)
                // ky = -1
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(&input[((y-1) * W + (x-1)) * C + c]), _mm512_loadu_ps(&weight[(0 * 3 + 0) * C + c]), sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(&input[((y-1) * W + (x  )) * C + c]), _mm512_loadu_ps(&weight[(0 * 3 + 1) * C + c]), sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(&input[((y-1) * W + (x+1)) * C + c]), _mm512_loadu_ps(&weight[(0 * 3 + 2) * C + c]), sum);
                
                // ky = 0
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(&input[((y  ) * W + (x-1)) * C + c]), _mm512_loadu_ps(&weight[(1 * 3 + 0) * C + c]), sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(&input[((y  ) * W + (x  )) * C + c]), _mm512_loadu_ps(&weight[(1 * 3 + 1) * C + c]), sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(&input[((y  ) * W + (x+1)) * C + c]), _mm512_loadu_ps(&weight[(1 * 3 + 2) * C + c]), sum);
                
                // ky = +1
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(&input[((y+1) * W + (x-1)) * C + c]), _mm512_loadu_ps(&weight[(2 * 3 + 0) * C + c]), sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(&input[((y+1) * W + (x  )) * C + c]), _mm512_loadu_ps(&weight[(2 * 3 + 1) * C + c]), sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(&input[((y+1) * W + (x+1)) * C + c]), _mm512_loadu_ps(&weight[(2 * 3 + 2) * C + c]), sum);
                
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
    // 2D-SSM is just 1D-SSM over the flattened sequence H*W for a given channel dimension.
    // For true vision parity, we would scan multiple directions, but structurally it
    // uses the exact same `ssm_step` backbone.
    nca::layers::SSMConfig ssm_cfg;
    ssm_cfg.d_inner = cfg.H * cfg.W * cfg.C;
    ssm_cfg.d_state = 16;
    
    nca::layers::ssm_step(h, A, B, C_proj, x, y, ssm_cfg);
}

} // namespace nca::vision
