#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace nca::centaur {

// ============================================================================
// CCSE — Token-Free Continuous Byte Output Head
// centaur/byte_output_head.hpp
//
// Implements "Token-Free" Continuous Output / Byte-level prediction.
// Bypasses massive O(V * d) Softmax bottlenecks (V=128k+) found in standard LLMs.
// Replaces it with a cache-resident 256-dimensional raw byte emitter that
// perfectly aligns with AVX-512 vector pipelines.
// ============================================================================

class ByteOutputHead {
public:
    ByteOutputHead(size_t d_model) : d_model_(d_model) {
        if (d_model % 16 != 0) {
            throw std::runtime_error("ByteOutputHead requires d_model % 16 == 0 for AVX-512");
        }
        
        // W_byte is 256 x d_model
        size_t bytes = 256 * d_model_ * sizeof(float);
#ifdef _WIN32
        w_byte_ = static_cast<float*>(_aligned_malloc(bytes, 64));
#else
        w_byte_ = static_cast<float*>(aligned_alloc(64, bytes));
#endif
        std::memset(w_byte_, 0, bytes);
        // Initialization can be done by a weight loader...
    }

    ~ByteOutputHead() {
        if (w_byte_) {
#ifdef _WIN32
            _aligned_free(w_byte_);
#else
            free(w_byte_);
#endif
        }
    }

    // Direct mapping from hidden state -> raw byte (tokenless)
    uint8_t generate_byte(const float* __restrict hidden_state) const {
        alignas(64) float logits[256];
        
        // 1. O(256 * D) Projection into 256 Byte space
        // Fits entirely in L1 cache, no V=100k memory thrashing
#if defined(__AVX512F__)
        for (int b = 0; b < 256; ++b) {
            __m512 v_sum = _mm512_setzero_ps();
            const float* w_row = &w_byte_[b * d_model_];
            for (size_t d = 0; d < d_model_; d += 16) {
                __m512 v_x = _mm512_loadu_ps(&hidden_state[d]);
                __m512 v_w = _mm512_load_ps(&w_row[d]);
                v_sum = _mm512_fmadd_ps(v_x, v_w, v_sum);
            }
            logits[b] = _mm512_reduce_add_ps(v_sum);
        }
#else
        for (int b = 0; b < 256; ++b) {
            float sum = 0.0f;
            for (size_t d = 0; d < d_model_; ++d) {
                sum += hidden_state[d] * w_byte_[b * d_model_ + d];
            }
            logits[b] = sum;
        }
#endif

        // 2. Argmax to find the predicted byte directly
        // No SentencePiece/BPE tokenizer needed!
        float max_val = logits[0];
        uint8_t best_byte = 0;
        
        for (int b = 1; b < 256; ++b) {
            if (logits[b] > max_val) {
                max_val = logits[b];
                best_byte = static_cast<uint8_t>(b);
            }
        }

        return best_byte;
    }

private:
    size_t d_model_;
    float* w_byte_ = nullptr;
};

} // namespace nca::centaur
