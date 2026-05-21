#pragma once
// ============================================================================
// NCA -- Sparse Local Attention (SLA) Layer
// core/layers/sla.hpp
//
// Sliding-window attention with W=256. For each query position, attends
// only to the last W keys — O(W) per token instead of O(N).
// In autoregressive mode: one query vector vs. the KV cache.
// ============================================================================

#include <cstddef>

namespace nca::layers {

struct SLAConfig {
    size_t d_head  = 128;   // Dimension per attention head
    size_t window  = 256;   // Sliding window size W
    size_t kv_len  = 256;   // Current number of valid KV entries (<= window)
};

// Single-head autoregressive attention step:
//   q:        query vector [d_head]
//   k_cache:  key cache    [window, d_head]  (ring buffer, row-major)
//   v_cache:  value cache  [window, d_head]  (ring buffer, row-major)
//   out:      output vector [d_head]
//   scores:   scratch buffer [window] (caller-allocated)
//
// Computes:  scores[j] = q · k_cache[j] / sqrt(d_head)
//            softmax(scores)
//            out = sum(scores[j] * v_cache[j])
void sla_step(
    const float* __restrict q,
    const float* __restrict k_cache,
    const float* __restrict v_cache,
    float* __restrict out,
    float* __restrict scores,
    SLAConfig cfg
);

} // namespace nca::layers
