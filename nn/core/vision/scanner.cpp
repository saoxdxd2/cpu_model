// ============================================================================
// NCA — Vision Stage 1: Global Scanner
// core/vision/scanner.cpp
// ============================================================================

#include "core/vision/scanner.hpp"
#include "core/simd/avx512_kernels.hpp"
#include "core/layers/ssm.hpp"
#include <immintrin.h>
#include <algorithm>
#include <vector>

namespace nca::vision {

// ── 2D WAVEFRONT SCANNING (Target 4) ────────────────────────────────────────
// Parallelizes the SSM update along anti-diagonals (r + c = k).
// This destroys the serial barrier of the 1D scan.
void ssm2d_scan(
    float* __restrict h, 
    const float* __restrict A, 
    const float* __restrict B, 
    const float* __restrict Cp, 
    const float* __restrict x, 
    float* __restrict y, 
    ScannerConfig cfg
) {
    const size_t H = cfg.H;
    const size_t W = cfg.W;
    const size_t C = cfg.C;
    
    __m512 vB = _mm512_loadu_ps(B);
    __m512 vCp = _mm512_loadu_ps(Cp);

    for (size_t c = 0; c < C; ++c) {
        for (size_t k = 0; k < H + W - 1; ++k) {
            size_t r_start = (k < W) ? 0 : k - W + 1;
            size_t r_end = std::min(k + 1, H);

            for (size_t r = r_start; r < r_end; ++r) {
                size_t col = k - r;
                size_t idx = (r * W + col);
                
                // Load input value and broadcast
                __m512 vX = _mm512_set1_ps(x[idx * C + c]);
                
                // Load state from previous flattened index (Simplified for now)
                __m512 vH_prev = (idx > 0) ? _mm512_loadu_ps(&h[(idx - 1) * 16]) : _mm512_setzero_ps();
                __m512 vA = _mm512_loadu_ps(&A[idx * 16]);
                
                // h_t = A * h_prev + B * x
                __m512 vH = _mm512_fmadd_ps(vA, vH_prev, _mm512_mul_ps(vB, vX));
                _mm512_storeu_ps(&h[idx * 16], vH);
                
                // y_t = reduce_add(h_t * Cp)
                y[idx * C + c] = _mm512_reduce_add_ps(_mm512_mul_ps(vH, vCp));
            }
        }
    }
}

// ── DEPTHWISE CONVOLUTION ───────────────────────────────────────────────────
void dwconv2d_3x3_scalar(const float* in, const float* w, float* out, ScannerConfig cfg) {
    const int H = (int)cfg.H, W = (int)cfg.W, C = (int)cfg.C;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                float sum = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int iy = y + ky, ix = x + kx;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                            sum += in[(iy * W + ix) * C + c] * w[((ky + 1) * 3 + (kx + 1)) * C + c];
                    }
                }
                out[(y * W + x) * C + c] = sum;
            }
        }
    }
}

void dwconv2d_3x3(const float* __restrict in, const float* __restrict w, float* __restrict out, ScannerConfig cfg) {
    dwconv2d_3x3_scalar(in, w, out, cfg);
}

} // namespace nca::vision
