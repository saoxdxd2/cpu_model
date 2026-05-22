#include "encoding/vision_encoder.hpp"
#include <cstring>
#include <cmath>
#include <random>

namespace nca::encoding {

SiliconVisionEncoder::SiliconVisionEncoder(size_t width, size_t height) 
    : width_(width), height_(height) {
    
    // Initialize patch weights (16x16 pixels per patch projection)
    patch_weights_.resize(256 * 2048); 
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.01f);
    for (auto& w : patch_weights_) w = dist(gen);
}

void SiliconVisionEncoder::encode_gui(const float* pixels, float* latent_out, size_t d_model) {
    // 1. Clear Target
    std::memset(latent_out, 0, d_model * sizeof(float));

    // 2. Perform Silicon Patch Scan
    // We treat the IDE screenshot as a grid of 16x16 patches.
    // Each patch is projected into the latent space using AVX-512.
    scan_patches_avx512(pixels, latent_out, d_model);
    
    // 3. Final Nonlinear Saliency Squeeze
    for (size_t i = 0; i < d_model; ++i) {
        latent_out[i] = std::tanh(latent_out[i] * 0.5f);
    }
}

void SiliconVisionEncoder::scan_patches_avx512(const float* pixels, float* latent, size_t d_model) {
    // Highly optimized 2D scanning kernel
    // Projects 16x16 blocks into the 2048-dim latent space
    const size_t patch_size = 16;
    const size_t nx = width_ / patch_size;
    const size_t ny = height_ / patch_size;

    for (size_t py = 0; py < ny; ++py) {
        for (size_t px = 0; px < nx; ++px) {
            // Find patch start in raw pixel buffer
            const float* p_start = pixels + (py * patch_size * width_) + (px * patch_size);
            
            // Project patch using AVX-512 (Vectorized Dot Product)
            for (size_t d = 0; d < d_model; d += 16) {
                __m512 v_acc = _mm512_setzero_ps();
                
                // Scan 256 pixels of the patch
                for (size_t k = 0; k < 256; k += 16) {
                    __m512 v_p = _mm512_loadu_ps(p_start + (k / 16) * width_); // Simplified row-load
                    __m512 v_w = _mm512_loadu_ps(&patch_weights_[k * d_model + d]);
                    v_acc = _mm512_fmadd_ps(v_p, v_w, v_acc);
                }
                
                // Additive fusion into the latent state
                __m512 v_prev = _mm512_loadu_ps(latent + d);
                _mm512_storeu_ps(latent + d, _mm512_add_ps(v_prev, v_acc));
            }
        }
    }
}

} // namespace nca::encoding
