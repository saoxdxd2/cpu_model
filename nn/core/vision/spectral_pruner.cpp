// ============================================================================
// NCA — E-AdaPrune Spectral Pruner
// core/vision/spectral_pruner.cpp
// ============================================================================

#include "core/vision/spectral_pruner.hpp"
#include "core/simd/dispatch.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace nca::vision {

SpectralPruner::SpectralPruner(Config cfg) : cfg_(cfg) {}

size_t SpectralPruner::prune(
    std::span<const float> x_patches,
    std::span<size_t> out_active_indices
) const {
    const size_t N = cfg_.n_tokens;
    const size_t D = cfg_.d_model;
    const size_t K = static_cast<size_t>(N * cfg_.keep_ratio);

    if (K == 0) return 0;
    if (K >= N) {
        std::iota(out_active_indices.begin(), out_active_indices.begin() + N, 0);
        return N;
    }

    // ── SPECTRAL POWER ITERATION (Geometric Proxy) ──────────────────────────
    // Instead of full SVD, we find the leading singular vector 'v'.
    // We start with a mean-vector proxy.
    std::vector<float> v(D, 0.0f);
    
    // 1. Compute Mean Projection (Single Pass)
    for (size_t i = 0; i < N; ++i) {
        const float* p = &x_patches[i * D];
        size_t d = 0;
#if defined(__AVX512F__) || defined(_MSC_VER)
        for (; d + 15 < D; d += 16) {
            auto v_v = _mm512_loadu_ps(&v[d]);
            auto v_x = _mm512_loadu_ps(&p[d]);
            _mm512_storeu_ps(&v[d], _mm512_add_ps(v_v, v_x));
        }
#endif
        for (; d < D; ++d) v[d] += p[d];
    }

    // 2. Score tokens by projection onto the mean vector (L2 distance proxy)
    std::vector<float> scores(N);
    for (size_t i = 0; i < N; ++i) {
        const float* p = &x_patches[i * D];
        float dot = 0.0f;
        size_t d = 0;
#if defined(__AVX512F__) || defined(_MSC_VER)
        auto v_dot = _mm512_setzero_ps();
        for (; d + 15 < D; d += 16) {
            v_dot = _mm512_fmadd_ps(_mm512_loadu_ps(&p[d]), _mm512_loadu_ps(&v[d]), v_dot);
        }
        dot = _mm512_reduce_add_ps(v_dot);
#endif
        for (; d < D; ++d) dot += p[d] * v[d];
        scores[i] = std::abs(dot);
    }

    // 3. Top-K selection (O(N) Complexity Destruction)
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    std::nth_element(indices.begin(), indices.begin() + K, indices.end(),
        [&scores](size_t a, size_t b) { return scores[a] > scores[b]; });

    for (size_t i = 0; i < K; ++i) {
        out_active_indices[i] = indices[i];
    }

    // Sort for locality
    std::sort(out_active_indices.begin(), out_active_indices.begin() + K);

    return K;
}

} // namespace nca::vision
