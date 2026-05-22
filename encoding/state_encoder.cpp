#include "encoding/state_encoder.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace nca::encoding {

SiliconEncoder::SiliconEncoder() {}

void SiliconEncoder::encode(const float* raw_obs, float* latent_out, size_t d_model) {
    // 1. Initial Wipe
    std::memset(latent_out, 0, d_model * sizeof(float));

    // 2. Encode Entity Section (First 80 floats)
    encode_entities_avx512(raw_obs, latent_out, d_model);

    // 3. Scan Spatial Grids (Remaining 1536 floats)
    // 3 Grids of 32x16 each.
    scan_spatial_grid_avx512(raw_obs + 80, latent_out, d_model);
}

void SiliconEncoder::encode_entities_avx512(const float* entities, float* latent, size_t d_model) {
    // Entities: 16 groups of 5 (Health, AP, X, Y, Side)
    // We process using AVX-512 (16 floats at a time)
    for (size_t i = 0; i < 80; i += 16) {
        __m512 v_ent = _mm512_loadu_ps(entities + i);
        
        // Importance Pruning: Amplify vital signs (Health < 0.2 is high saliency)
        __m512 v_threshold = _mm512_set1_ps(0.2f);
        __mmask16 low_health = _mm512_cmp_ps_mask(v_ent, v_threshold, _CMP_LT_OS);
        
        // Boost low health signal to ensure engine notices casualties
        v_ent = _mm512_mask_mul_ps(v_ent, low_health, v_ent, _mm512_set1_ps(2.0f));
        
        // Direct injection into the start of the latent vector
        size_t target_idx = i % d_model;
        __m512 v_prev = _mm512_loadu_ps(latent + target_idx);
        _mm512_storeu_ps(latent + target_idx, _mm512_add_ps(v_prev, v_ent));
    }
}

void SiliconEncoder::scan_spatial_grid_avx512(const float* grid, float* latent, size_t d_model) {
    // 1536 floats. 1536 / 16 = 96 AVX-512 iterations.
    const __m512 v_noise_gate = _mm512_set1_ps(0.01f);
    
    for (size_t i = 0; i < 1536; i += 16) {
        __m512 v_data = _mm512_loadu_ps(grid + i);
        
        // Noise Gate: Zero out negligible signals
        __mmask16 mask = _mm512_cmp_ps_mask(v_data, v_noise_gate, _CMP_GT_OS);
        v_data = _mm512_maskz_mov_ps(mask, v_data);
        
        // Fold spatial data into the latent dimension
        // We use a modular fold to create a "Compressed Spatial Hash"
        size_t target_idx = (i + 128) % d_model; 
        
        __m512 v_target = _mm512_loadu_ps(latent + target_idx);
        _mm512_storeu_ps(latent + target_idx, _mm512_add_ps(v_target, v_data));
    }

    // Final RMS Normalization of the latent vector to ensure saturation stability
    float sum_sq = 0.0f;
    for(size_t i=0; i<d_model; ++i) sum_sq += latent[i] * latent[i];
    float scale = 1.0f / (std::sqrt(sum_sq / d_model) + 1e-6f);
    
    for(size_t i=0; i<d_model; i += 16) {
        __m512 v_l = _mm512_loadu_ps(latent + i);
        _mm512_storeu_ps(latent + i, _mm512_mul_ps(v_l, _mm512_set1_ps(scale)));
    }
}

} // namespace nca::encoding
