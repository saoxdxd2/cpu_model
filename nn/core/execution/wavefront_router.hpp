#pragma once
// ============================================================================
// NCA — Geometric Wavefront Router (v1.0 - AVX-512)
// core/execution/wavefront_router.hpp
// ============================================================================

#include "core/simd/memory.hpp"
#include <immintrin.h>
#include <vector>
#include <cstdint>

namespace nca::execution {

// ── 1. The Compressed 8-Byte Schema ──
struct alignas(8) GeometricBranch {
    uint32_t next_shape_id; // 4 bytes: The Explicit Structural Pointer
    uint16_t rule_mask;     // 2 bytes: Logical Rule / Gate Filter
    uint8_t  width;         // 1 byte:  Q-Value / Probability Bandwidth
    uint8_t  is_end;        // 1 byte:  Terminal Flag
};

// ── 2. Vectorized Randomness Generator ──
struct alignas(64) SimdRandomState {
    __m512i state;

    SimdRandomState() {
        uint32_t seeds[16];
        for(int i = 0; i < 16; ++i) seeds[i] = 1337 + i * 999;
        state = _mm512_loadu_si512(seeds);
    }

    inline __m512 generate_uniform() {
        __m512i x = state;
        x = _mm512_xor_si512(x, _mm512_slli_epi32(x, 13));
        x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 17));
        x = _mm512_xor_si512(x, _mm512_slli_epi32(x, 5));
        state = x;

        __m512i masked = _mm512_and_si512(x, _mm512_set1_epi32(0x007FFFFF));
        __m512i float_bits = _mm512_or_si512(masked, _mm512_set1_epi32(0x3F800000));
        return _mm512_sub_ps(_mm512_castsi512_ps(float_bits), _mm512_set1_ps(1.0f));
    }
};

// ── 3. The Geometric Execution Core ──
class WavefrontRouter {
public:
    WavefrontRouter(size_t n_concepts, size_t wavefront_width = 16);
    ~WavefrontRouter() = default;

    // Load compiled .geo graphs (Replaces Dense Weight Loading)
    void load_geometric_graph(const std::vector<std::vector<GeometricBranch>>& compiled_graph);
    
    // Initialize a default graph if no graph is loaded
    void initialize_default_graph();

    // Execute Stochastic Pointer Chasing
    // Tracks up to `wavefront_width` parallel realities using _mm512_i32gather_epi32
    void step_wavefront(float* state, float temperature = 0.2f);

private:
    size_t n_concepts_;
    size_t wavefront_width_;
    SimdRandomState rng_;

    // Flattened Structure of Arrays (SoA) for Gather Instructions
    ::nca::simd::aligned_unique_ptr<uint32_t[]> flat_pointers_;
    ::nca::simd::aligned_unique_ptr<float[]>    flat_probs_;
    ::nca::simd::aligned_unique_ptr<uint32_t[]> graph_offsets_;
    size_t total_branches_ = 0;
};

} // namespace nca::execution
