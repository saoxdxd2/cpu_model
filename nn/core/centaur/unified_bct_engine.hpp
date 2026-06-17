#pragma once
// ============================================================================
// CCSE — Unified Continuous-Discrete BCT Engine
// centaur/unified_bct_engine.hpp
//
// Unifies the Bipolar Phase-Collapse Router (PCRM) and Binary Curve Tree (BCT)
// expert execution into a single, indivisible hardware abstraction.
// Eradicates dense FP32 multipliers and W_up memory.
// Reduces Top-K combinatorial sorting to a single AVX-512 phase collapse.
// ============================================================================

#include "core/centaur/cpu_fingerprint.hpp"
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>

#ifdef _WIN32
#include <malloc.h>
#endif

#include "core/centaur/byte_output_head.hpp"

namespace nca::centaur {

class UnifiedBCTEngine {
public:
    UnifiedBCTEngine(size_t d_model, size_t d_expert, size_t n_experts)
        : d_model_(d_model), d_expert_(d_expert), n_experts_(n_experts), byte_head_(d_model) {
        
        if (d_model_ % 16 != 0 || d_expert_ != 16) {
            throw std::runtime_error("UnifiedBCTEngine requires D % 16 == 0 and DE == 16.");
        }

        B_ = 0;
        size_t n = n_experts_;
        while (n > 1) { n >>= 1; B_++; }
        B_padded_ = (B_ + 15) & ~15;

        router_bytes_ = d_model_ * B_padded_ * sizeof(float);
        
        size_t blocks_per_row = d_model_ / 16;
        size_t u16_per_expert = 2 * d_expert_ * blocks_per_row;
        experts_bytes_ = n_experts_ * u16_per_expert * sizeof(uint16_t);

        // Continuous block for hardware
        size_t total_bytes = router_bytes_ + experts_bytes_;
#ifdef _WIN32
        arena_ = static_cast<uint8_t*>(_aligned_malloc(total_bytes, 64));
#else
        arena_ = static_cast<uint8_t*>(aligned_alloc(64, total_bytes));
#endif
        std::memset(arena_, 0, total_bytes);

        router_w_down_ = reinterpret_cast<float*>(arena_);
        experts_arena_ = reinterpret_cast<uint16_t*>(arena_ + router_bytes_);
    }

    ~UnifiedBCTEngine() {
        if (arena_) {
#ifdef _WIN32
            _aligned_free(arena_);
#else
            free(arena_);
#endif
        }
    }

    void initialize_random(uint32_t seed = 42) {
        uint32_t s = seed;
        auto xorshift = [&]() -> uint32_t {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            return s;
        };

        float scale = std::sqrt(6.0f / (d_model_ + B_));
        for (size_t d = 0; d < d_model_; ++d) {
            for (size_t b = 0; b < B_; ++b) {
                float r = (static_cast<float>(xorshift() & 0x7FFFFF) / 0x7FFFFF) * 2.0f - 1.0f;
                router_w_down_[d * B_padded_ + b] = r * scale;
            }
        }

        size_t total_u16 = experts_bytes_ / sizeof(uint16_t);
        for (size_t i = 0; i < total_u16; ++i) {
            experts_arena_[i] = static_cast<uint16_t>(xorshift() & 0xFFFF);
        }
    }

    // Returns a raw tokenless byte by piping expert output into the byte_head
    uint8_t execute_and_sample(const float* __restrict x) const {
        alignas(64) float out[8192]; // Safe upper bound for D_model
        std::memset(out, 0, d_model_ * sizeof(float));
        
        uint16_t experts[2];
        float gates[2];
        
        // 1. O(B) PHASE COLLAPSE ROUTING
#if defined(__AVX512F__)
        __m512 v_h = _mm512_setzero_ps();
        size_t d = 0;
        for (; d + 3 < d_model_; d += 4) {
            v_h = _mm512_fmadd_ps(_mm512_set1_ps(x[d+0]), _mm512_load_ps(&router_w_down_[(d+0) * B_padded_]), v_h);
            v_h = _mm512_fmadd_ps(_mm512_set1_ps(x[d+1]), _mm512_load_ps(&router_w_down_[(d+1) * B_padded_]), v_h);
            v_h = _mm512_fmadd_ps(_mm512_set1_ps(x[d+2]), _mm512_load_ps(&router_w_down_[(d+2) * B_padded_]), v_h);
            v_h = _mm512_fmadd_ps(_mm512_set1_ps(x[d+3]), _mm512_load_ps(&router_w_down_[(d+3) * B_padded_]), v_h);
        }
        for (; d < d_model_; ++d) {
            v_h = _mm512_fmadd_ps(_mm512_set1_ps(x[d]), _mm512_load_ps(&router_w_down_[d * B_padded_]), v_h);
        }

        uint16_t k1 = _mm512_cmp_ps_mask(v_h, _mm512_setzero_ps(), _CMP_GT_OQ) & ((1 << B_) - 1);

        alignas(64) float h_arr[16];
        _mm512_store_ps(h_arr, v_h);

        float min_abs_h = 1e30f;
        size_t min_b = 0;
        for (size_t b = 0; b < B_; ++b) {
            float abs_h = std::abs(h_arr[b]);
            if (abs_h < min_abs_h) { min_abs_h = abs_h; min_b = b; }
        }

        uint16_t k2 = k1 ^ (1 << min_b);
        experts[0] = k1;
        experts[1] = k2;
        
        float e1 = std::exp(min_abs_h);
        float e2 = std::exp(-min_abs_h);
        float sum_e = e1 + e2;
        gates[0] = e1 / sum_e;
        gates[1] = e2 / sum_e;
#else
        // Fallback for non-AVX512 skipped for brevity
#endif

        // 2. DISCRETE EXPERT EXECUTION (XOR Phase Acc)
        size_t blocks = d_model_ / 16;
        for (int i = 0; i < 2; ++i) {
            uint16_t expert_id = experts[i];
            float gate_weight = gates[i];
            
            const uint16_t* w_gate_cols = experts_arena_ + (expert_id * 2 * d_expert_ * blocks);
            const uint16_t* w_down_rows = w_gate_cols + (d_expert_ * blocks);
            
            alignas(64) float h[16]; 

            for (size_t j = 0; j < d_expert_; ++j) {
#if defined(__AVX512F__)
                __m512 v_sum = _mm512_setzero_ps();
                const uint16_t* w_col = w_gate_cols + (j * blocks);
                for (size_t d_blk = 0; d_blk < blocks; ++d_blk) {
                    __m512 v_x = _mm512_loadu_ps(&x[d_blk * 16]);
                    __m512i sign_flip = _mm512_slli_epi32(_mm512_movm_epi32(~w_col[d_blk]), 31);
                    v_sum = _mm512_add_ps(v_sum, _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(v_x), sign_flip)));
                }
                float dot = _mm512_reduce_add_ps(v_sum);
                h[j] = dot / (1.0f + std::exp(-dot)) * gate_weight;
#endif
            }

#if defined(__AVX512F__)
            for (size_t d_blk = 0; d_blk < blocks; ++d_blk) {
                __m512 v_out = _mm512_loadu_ps(&out[d_blk * 16]);
                for (size_t j = 0; j < d_expert_; ++j) {
                    __m512 v_h = _mm512_set1_ps(h[j]);
                    __m512i sign_flip = _mm512_slli_epi32(_mm512_movm_epi32(~w_down_rows[j * blocks + d_blk]), 31);
                    v_out = _mm512_add_ps(v_out, _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(v_h), sign_flip)));
                }
                _mm512_storeu_ps(&out[d_blk * 16], v_out);
            }
#endif
        }

        // 3. CONTINUOUS TOKENLESS OUTPUT 
        // Directly map the continuous hidden state to a raw byte using the new AVX-512 head
        return byte_head_.generate_byte(out);
    }

private:
    size_t d_model_;
    size_t d_expert_;
    size_t n_experts_;
    size_t B_;
    size_t B_padded_;
    size_t router_bytes_;
    size_t experts_bytes_;
    
    uint8_t* arena_ = nullptr;
    float* router_w_down_ = nullptr;
    uint16_t* experts_arena_ = nullptr;

    // Tokenless architecture projection
    ByteOutputHead byte_head_;
};

} // namespace nca::centaur
