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
    
#if defined(__AVX512F__) || defined(_MSC_VER)
    __m512 vB = _mm512_loadu_ps(B);
    __m512 vCp = _mm512_loadu_ps(Cp);

    for (size_t c = 0; c < C; ++c) {
        for (size_t k = 0; k < H + W - 1; ++k) {
            size_t r_start = (k < W) ? 0 : k - W + 1;
            size_t r_end = std::min(k + 1, H);

            for (size_t r = r_start; r < r_end; ++r) {
                size_t col = k - r;
                size_t idx = (r * W + col);
                
                __m512 vX = _mm512_set1_ps(x[idx * C + c]);
                __m512 vA = _mm512_loadu_ps(&A[idx * 16]);
                
                // Branchless state load: use a dummy zero buffer for idx == 0
                alignas(64) static const float zero_buf[16] = {0};
                const float* p_prev = (idx > 0) ? &h[(idx - 1) * 16] : zero_buf;
                __m512 vH_prev = _mm512_load_ps(p_prev); // Must use load_ps since zero_buf is aligned
                if (idx > 0) vH_prev = _mm512_loadu_ps(p_prev);
                
                __m512 vH = _mm512_fmadd_ps(vA, vH_prev, _mm512_mul_ps(vB, vX));
                _mm512_storeu_ps(&h[idx * 16], vH);
                
                y[idx * C + c] = _mm512_reduce_add_ps(_mm512_mul_ps(vH, vCp));
            }
        }
    }
#else
    __m256 vB0 = _mm256_loadu_ps(B);
    __m256 vB1 = _mm256_loadu_ps(B + 8);
    __m256 vCp0 = _mm256_loadu_ps(Cp);
    __m256 vCp1 = _mm256_loadu_ps(Cp + 8);

    for (size_t c = 0; c < C; ++c) {
        for (size_t k = 0; k < H + W - 1; ++k) {
            size_t r_start = (k < W) ? 0 : k - W + 1;
            size_t r_end = std::min(k + 1, H);

            for (size_t r = r_start; r < r_end; ++r) {
                size_t col = k - r;
                size_t idx = (r * W + col);
                
                __m256 vX = _mm256_set1_ps(x[idx * C + c]);
                
                alignas(32) static const float zero_buf[16] = {0};
                const float* p_prev0 = (idx > 0) ? &h[(idx - 1) * 16] : zero_buf;
                const float* p_prev1 = (idx > 0) ? &h[(idx - 1) * 16 + 8] : zero_buf + 8;
                
                __m256 vHp0 = _mm256_load_ps(p_prev0);
                if(idx > 0) vHp0 = _mm256_loadu_ps(p_prev0);
                __m256 vHp1 = _mm256_load_ps(p_prev1);
                if(idx > 0) vHp1 = _mm256_loadu_ps(p_prev1);
                
                __m256 vA0 = _mm256_loadu_ps(&A[idx * 16]);
                __m256 vA1 = _mm256_loadu_ps(&A[idx * 16 + 8]);
                
                __m256 vH0 = _mm256_fmadd_ps(vA0, vHp0, _mm256_mul_ps(vB0, vX));
                __m256 vH1 = _mm256_fmadd_ps(vA1, vHp1, _mm256_mul_ps(vB1, vX));
                
                _mm256_storeu_ps(&h[idx * 16], vH0);
                _mm256_storeu_ps(&h[idx * 16 + 8], vH1);
                
                __m256 sum_vec = _mm256_add_ps(_mm256_mul_ps(vH0, vCp0), _mm256_mul_ps(vH1, vCp1));
                __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                sum_low = _mm_add_ps(sum_low, sum_high);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                y[idx * C + c] = _mm_cvtss_f32(sum_low);
            }
        }
    }
#endif
}

// ── DEPTHWISE CONVOLUTION ───────────────────────────────────────────────────
void dwconv2d_3x3(const float* __restrict in, const float* __restrict w, float* __restrict out, ScannerConfig cfg) {
    const int H = (int)cfg.H, W = (int)cfg.W, C = (int)cfg.C;
    
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int c = 0;
#if defined(__AVX512F__) || defined(_MSC_VER)
            for (; c + 15 < C; c += 16) {
                __m512 sum = _mm512_setzero_ps();
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int iy = y + ky, ix = x + kx;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                            __m512 v_in = _mm512_loadu_ps(&in[(iy * W + ix) * C + c]);
                            __m512 v_w = _mm512_loadu_ps(&w[((ky + 1) * 3 + (kx + 1)) * C + c]);
                            sum = _mm512_fmadd_ps(v_in, v_w, sum);
                        }
                    }
                }
                _mm512_storeu_ps(&out[(y * W + x) * C + c], sum);
            }
#else
            for (; c + 7 < C; c += 8) {
                __m256 sum = _mm256_setzero_ps();
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int iy = y + ky, ix = x + kx;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                            __m256 v_in = _mm256_loadu_ps(&in[(iy * W + ix) * C + c]);
                            __m256 v_w = _mm256_loadu_ps(&w[((ky + 1) * 3 + (kx + 1)) * C + c]);
                            sum = _mm256_fmadd_ps(v_in, v_w, sum);
                        }
                    }
                }
                _mm256_storeu_ps(&out[(y * W + x) * C + c], sum);
            }
#endif
            for (; c < C; ++c) {
                float sum = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int iy = y + ky, ix = x + kx;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                            sum += in[(iy * W + ix) * C + c] * w[((ky + 1) * 3 + (kx + 1)) * C + c];
                        }
                    }
                }
                out[(y * W + x) * C + c] = sum;
            }
        }
    }
}

} // namespace nca::vision

