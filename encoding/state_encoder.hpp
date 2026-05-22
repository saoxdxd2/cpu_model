#pragma once
#include <cstddef>
#include <immintrin.h>

namespace nca::encoding {

/**
 * SiliconEncoder
 * High-performance state transformation using AVX-512.
 * Performs spatial scanning and importance pruning on tactical grids.
 */
class SiliconEncoder {
public:
    SiliconEncoder();

    /**
     * Encodes a raw 1616-dim tactical observation into a D_MODEL latent vector.
     * Uses AVX-512 Scan+Prune logic to filter out noise and emphasize threats.
     */
    void encode(const float* raw_obs, float* latent_out, size_t d_model);

private:
    // SIMD Scanning Kernels
    static void scan_spatial_grid_avx512(const float* grid, float* latent, size_t d_model);
    static void encode_entities_avx512(const float* entities, float* latent, size_t d_model);
};

} // namespace nca::encoding
