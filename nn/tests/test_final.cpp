// ============================================================================
// NCA — Final Integrated Verification (Golden Reference Validation)
// tests/test_final.cpp
// ============================================================================

#include "core/execution/multimodal_engine.hpp"
#include "config/model_config.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>

// FNV-1a Hash for Golden Reference
inline uint64_t fnv1a_hash(const float* data, size_t n) {
    uint64_t hash = 14695981039346656037ULL;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

int main() {
    std::cout << "+------------------------------------------+\n";
    std::cout << "|  NCA -- Final Integrated Verification    |\n";
    std::cout << "+------------------------------------------+\n\n";

    const size_t D = nca::config::D_MODEL;
    alignas(64) float text_in[2048];
    alignas(64) float img_in[16 * 16 * 128];
    alignas(64) float out[2048];

    for(int i=0; i<D; ++i) text_in[i] = (float)i / D;
    for(int i=0; i<16*16*128; ++i) img_in[i] = (float)(i % 100) / 100.0f;

    uint64_t hash_v27 = 0;
    double ms_v27 = 0;

    // ── 1. VERIFY UNIFIED SDMS_PREDICTIVE ENGINE (V27.0) ───────────────────
    {
        std::cout << "  [RUN ] Backend: SDMS_Predictive (Unified v27.0)... " << std::flush;
        nca::config::EngineConfig cfg;
        cfg.logic_backend = nca::config::LogicBackend::SDMS_Predictive;
        nca::execution::MultimodalEngine engine(cfg);
        auto t0 = std::chrono::high_resolution_clock::now();
        engine.step(text_in, img_in, out);
        auto t1 = std::chrono::high_resolution_clock::now();
        ms_v27 = std::chrono::duration<double, std::milli>(t1 - t0).count();
        hash_v27 = fnv1a_hash(out, D);
        
        const uint64_t GOLDEN_V27 = 0xd18504447fedc325ULL;
        if (hash_v27 == GOLDEN_V27) {
            std::cout << "OK (Hash Match: 0x" << std::hex << hash_v27 << std::dec << ", " << ms_v27 << " ms)\n";
        } else {
            std::cout << "FAIL (Expected: 0x" << std::hex << GOLDEN_V27 << " Found: 0x" << hash_v27 << std::dec << ")\n";
            return 1;
        }
    }

    std::cout << "\n  [PASS] Unified Logic Backend Verified and Performance-Hardened.\n";
    return 0;
}

