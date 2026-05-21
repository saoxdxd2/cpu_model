#pragma once
// ============================================================================
// NCA — Saliency and Importance Classifier
// core/execution/importance.hpp
// ============================================================================

#include "config/model_config.hpp"
#include <vector>

namespace nca::execution {

struct ImportanceDecision {
    float saliency;      // 0..1 (Information entropy)
    float novelty;       // distance from internal state
    float surprise;      // prediction error
    bool  should_learn;  // memory-write gate
    int   act_cycles;    // dynamic depth
    bool  is_fact;       // high-fidelity fact?
};

class ImportanceClassifier {
public:
    ImportanceDecision classify(
        const float* x, 
        const float* state, 
        const float* prediction, 
        size_t d_model
    ) const;
};

} // namespace nca::execution
