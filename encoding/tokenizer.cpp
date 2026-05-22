#include "encoding/tokenizer.hpp"
#include <immintrin.h>
#include <random>
#include <cmath>

namespace nca::encoding {

SaliencyTokenizer::SaliencyTokenizer(size_t vocab_size, size_t d_model) 
    : vocab_size_(vocab_size), d_model_(d_model) {
    
    codebook_.resize(vocab_size_ * d_model_);
    
    // Initialize Codebook with high-saliency noise (Unit Sphere)
    std::mt19937 gen(1337);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < vocab_size_; ++i) {
        float sum_sq = 0.0f;
        for (size_t j = 0; j < d_model_; ++j) {
            float val = dist(gen);
            codebook_[i * d_model_ + j] = val;
            sum_sq += val * val;
        }
        float inv_norm = 1.0f / std::sqrt(sum_sq + 1e-8f);
        for (size_t j = 0; j < d_model_; ++j) {
            codebook_[i * d_model_ + j] *= inv_norm;
        }
    }
}

void SaliencyTokenizer::tokenize(const float* latent, std::vector<uint32_t>& tokens_out) {
    // We'll find the Top-K best matching codebook vectors
    // Using AVX-512 to accelerate the dot product search
    tokens_out.clear();
    
    float best_score = -1e9f;
    uint32_t best_idx = 0;

    for (size_t i = 0; i < vocab_size_; ++i) {
        const float* cb_ptr = &codebook_[i * d_model_];
        
        // AVX-512 Dot Product
        __m512 v_sum = _mm512_setzero_ps();
        for (size_t j = 0; j < d_model_; j += 16) {
            __m512 v_l = _mm512_loadu_ps(latent + j);
            __m512 v_c = _mm512_loadu_ps(cb_ptr + j);
            v_sum = _mm512_fmadd_ps(v_l, v_c, v_sum);
        }
        
        float score = _mm512_reduce_add_ps(v_sum);
        
        if (score > best_score) {
            best_score = score;
            best_idx = static_cast<uint32_t>(i);
        }
    }

    tokens_out.push_back(best_idx);
}

const float* SaliencyTokenizer::get_embedding(uint32_t token_id) const {
    if (token_id >= vocab_size_) return nullptr;
    return &codebook_[token_id * d_model_];
}

} // namespace nca::encoding
