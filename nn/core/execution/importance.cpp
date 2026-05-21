// ============================================================================
// NCA — Saliency and Importance Classifier Implementation (v7.0 - Sensitive)
// core/execution/importance.cpp
// ============================================================================

#include "core/execution/importance.hpp"
#include <cmath>
#include <algorithm>

namespace nca::execution {

ImportanceDecision ImportanceClassifier::classify(
    const float* x, 
    const float* state, 
    const float* prediction, 
    size_t d_model
) const {
    ImportanceDecision d;
    
    // 1. Saliency (Variance)
    double mean = 0, sq_mean = 0;
    for(size_t i=0; i<d_model; ++i) { mean += x[i]; sq_mean += x[i]*x[i]; }
    d.saliency = (float)((sq_mean/d_model) - (mean/d_model)*(mean/d_model));

    // 2. Novelty (Contextual Distance)
    double dot_s = 0, norm_x = 0, norm_s = 0;
    for(size_t i=0; i<d_model; ++i) {
        dot_s += x[i] * state[i];
        norm_x += x[i] * x[i];
        norm_s += state[i] * state[i];
    }
    float sim_s = (norm_x > 1e-6 && norm_s > 1e-6) ? (float)(dot_s / (std::sqrt(norm_x) * std::sqrt(norm_s))) : 0.0f;
    d.novelty = 1.0f - std::abs(sim_s);

    // 3. Decision Logic (v7.0)
    // A signal is a Fact if it has ANY structure (Saliency > 0.01) AND is novel (Novelty > 0.2).
    // This catches the 25% Trigger Key.
    d.is_fact = (d.saliency > 0.01f && d.novelty > 0.2f);
    
    // We only learn if it's a strong, novel fact.
    d.should_learn = (d.saliency > 0.1f && d.novelty > 0.5f);
    
    d.act_cycles = d.is_fact ? 64 : 1;

    return d;
}

} // namespace nca::execution
