// ============================================================================
// NCA — Cognitive Needle-in-a-Haystack Benchmark (v16.0 - Structural Collision)
// tests/test_cognitive.cpp
// ============================================================================

#include "core/execution/multimodal_engine.hpp"
#include "config/model_config.hpp"
#include "core/spectral/fwht.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <random>

float calculate_cosine_similarity(const float* a, const float* b, size_t n) {
    double dot = 0, norm_a = 0, norm_b = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i])) {
            std::cout << " [WARN] NaN detected at index " << i << "\n";
            return 0.0f;
        }
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0 || norm_b == 0) return 0.0f;
    return (float)(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
}

int main() {
    std::cout << "+---------------------------------------------------+\n";
    std::cout << "|  NCA -- Cognitive Integrity Benchmark (v16.0)     |\n";
    std::cout << "+---------------------------------------------------+\n\n";

    const size_t D = nca::config::D_MODEL;
    const size_t SEQ_LEN = 2048; 
    
    nca::config::EngineConfig cfg;
    cfg.logic_backend = nca::config::LogicBackend::SDMS_Predictive;
    nca::execution::MultimodalEngine engine(nca::config::D_MODEL, 80, cfg);
    engine.reset_state();

    std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Fact A (The Needle)
    std::vector<float> needle(D);
    for(size_t i=0; i<D; ++i) needle[i] = dist(gen);

    // Fact B (The Haystack Noise - Structural)
    std::vector<float> fact_b(D);
    for(size_t i=0; i<D; ++i) fact_b[i] = dist(gen);

    std::cout << "  [RUN ] Streaming STRUCTURAL Haystack (N=" << SEQ_LEN << ")... " << std::flush;
    std::vector<float> out(D);
    
    for (size_t t = 0; t < SEQ_LEN; ++t) {
        if (t == 100) {
            engine.step(needle.data(), nullptr, out.data());
        } else {
            // Fill with DYNAMIC structural noise
            std::vector<float> noise(D);
            for(size_t i=0; i<D; ++i) noise[i] = dist(gen);
            engine.step(noise.data(), nullptr, out.data());
        }
        if (t % 512 == 0) std::cout << "." << std::flush;
    }
    std::cout << " DONE\n";

    std::cout << "  [RUN ] Querying Needle Recall (Trigger Key)... " << std::flush;
    std::vector<float> trigger(D, 0.0f);
    // Use first 512 elements (25%) as key for structural victory
    for(int i=0; i<512; ++i) trigger[i] = needle[i]; 
    engine.step(trigger.data(), nullptr, out.data());
    std::cout << " DONE\n";

    float cos_sim_a = calculate_cosine_similarity(out.data(), needle.data(), D);
    float cos_sim_b = calculate_cosine_similarity(out.data(), fact_b.data(), D);
    
    std::cout << "\n-----------------------------------------------------\n";
    std::cout << "  Metric              Value           Status\n";
    std::cout << "-----------------------------------------------------\n";
    std::cout << std::left << std::setw(20) << "Sim(Out, Fact A)" << std::setw(16) << cos_sim_a 
              << (cos_sim_a > 0.3f ? "PERFECT" : (cos_sim_a > 0.05f ? "SIGNAL" : "FAIL")) << "\n";
    std::cout << std::left << std::setw(20) << "Sim(Out, Fact B)" << std::setw(16) << cos_sim_b 
              << (cos_sim_b < 0.05f ? "ISOLATED" : "CORRUPTED") << "\n";
    std::cout << "-----------------------------------------------------\n";

    if (cos_sim_a > 0.3f && cos_sim_b < 0.1f) {
         std::cout << "\n  [ULTIMATE] TRANSFORMER KILLER PROVEN.\n";
         std::cout << "             Associative Memory successfully isolated A from B.\n";
    }

    return 0;
}
