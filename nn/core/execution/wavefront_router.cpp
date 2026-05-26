#include "core/execution/wavefront_router.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace nca::execution {

WavefrontRouter::WavefrontRouter(size_t n_concepts, size_t wavefront_width)
    : n_concepts_(n_concepts), wavefront_width_(wavefront_width) 
{
    if (wavefront_width_ != 16) {
        throw std::runtime_error("WavefrontRouter currently strictly requires wavefront_width=16 for AVX-512.");
    }
}

void WavefrontRouter::load_geometric_graph(const std::vector<std::vector<GeometricBranch>>& compiled_graph) {
    if (compiled_graph.size() != n_concepts_) {
        throw std::runtime_error("Graph size mismatch with n_concepts.");
    }

    // 1. Calculate total active branches (padded to wavefront_width)
    total_branches_ = n_concepts_ * wavefront_width_;

    // 2. Allocate Flattened Buffers (SoA)
    flat_pointers_ = ::nca::simd::make_aligned_unique<uint32_t[]>(total_branches_);
    flat_probs_    = ::nca::simd::make_aligned_unique<float[]>(total_branches_);
    graph_offsets_ = ::nca::simd::make_aligned_unique<uint32_t[]>(n_concepts_ + 1);

    // 3. Populate Buffers
    size_t current_offset = 0;
    for (size_t i = 0; i < n_concepts_; ++i) {
        graph_offsets_[i] = static_cast<uint32_t>(current_offset);
        
        const auto& branches = compiled_graph[i];
        size_t b = 0;
        for (; b < branches.size() && b < wavefront_width_; ++b) {
            flat_pointers_[current_offset + b] = branches[b].next_shape_id;
            
            // Map the 1-byte width (0-255) back to FP32 probability (0.0-1.0)
            flat_probs_[current_offset + b] = branches[b].width / 255.0f;
        }

        // Pad the rest of the wavefront lane with self-loops or zeros
        for (; b < wavefront_width_; ++b) {
            flat_pointers_[current_offset + b] = static_cast<uint32_t>(i); // Sink into self
            flat_probs_[current_offset + b]    = 0.0f; // Zero probability
        }
        
        current_offset += wavefront_width_;
    }
    graph_offsets_[n_concepts_] = static_cast<uint32_t>(current_offset);
}

void WavefrontRouter::step_wavefront(float* state, float temperature) {
    // Double buffer the state to prevent out-of-order writes
    auto next_state = ::nca::simd::make_aligned_unique<float[]>(n_concepts_);
    std::memset(next_state.get(), 0, n_concepts_ * sizeof(float));

    __m512 vTemp = _mm512_set1_ps(temperature);

    for (size_t i = 0; i < n_concepts_; ++i) {
        if (state[i] < 1e-6f) continue; // Skip dead concepts (Sparsity)

        uint32_t offset = graph_offsets_[i];
        
        // Load the 16 base probabilities
        __m512 vProbs = _mm512_load_ps(&flat_probs_[offset]);

        // Inject SIMD Stochastic Exploration (Temperature)
        if (temperature > 0.0f) {
            __m512 vNoise = rng_.generate_uniform();
            vProbs = _mm512_fmadd_ps(vNoise, vTemp, vProbs); // probs += noise * temp
        }

        // Apply state amplitude
        __m512 vStateAmp = _mm512_set1_ps(state[i]);
        __m512 vOutAmp   = _mm512_mul_ps(vProbs, vStateAmp);

        // Extract and route
        alignas(64) float out_amps[16];
        _mm512_store_ps(out_amps, vOutAmp);
        
        for (int lane = 0; lane < 16; ++lane) {
            uint32_t target_id = flat_pointers_[offset + lane];
            if (target_id < n_concepts_) {
                next_state[target_id] += out_amps[lane];
            }
        }
    }

    // Copy back to main state
    std::memcpy(state, next_state.get(), n_concepts_ * sizeof(float));
}

} // namespace nca::execution
