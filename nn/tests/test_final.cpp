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

    // Initialize inputs
    for(int i=0; i<D; ++i) text_in[i] = (float)i / D;
    for(int i=0; i<16*16*128; ++i) img_in[i] = (float)(i % 100) / 100.0f;

    // ── 1. VERIFY HASHED ROUTER (DEFAULT) ──────────────────────────────────
    {
        std::cout << "  [RUN ] Backend: HashedRouter (Default)... " << std::flush;
        nca::execution::MultimodalEngine engine;
        auto t0 = std::chrono::high_resolution_clock::now();
        engine.step(text_in, img_in, out);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        uint64_t h = fnv1a_hash(out, D);
        
        const uint64_t GOLDEN_H = 0x40612f1ed3b468ddULL;
        
        if (h == GOLDEN_H) {
            std::cout << "OK (Hash: 0x" << std::hex << h << std::dec << ", " << ms << " ms)\n";
        } else {
            std::cout << "FAIL (Found: 0x" << std::hex << h << ")\n";
            return 1;
        }
    }

    // ── 2. VERIFY SPECTRAL RLS (V7.0) ───────────────────────────────────────
    {
        std::cout << "  [RUN ] Backend: SpectralRLS (Opt-in)... " << std::flush;
        nca::config::EngineConfig cfg;
        cfg.logic_backend = nca::config::LogicBackend::SpectralRLS;
        nca::execution::MultimodalEngine engine(cfg);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        engine.step(text_in, img_in, out);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        uint64_t h = fnv1a_hash(out, D);
        
        const uint64_t GOLDEN_S = 0xad2ddafcd00a0dfdULL; 

        if (h == GOLDEN_S) {
             std::cout << "OK (Hash: 0x" << std::hex << h << std::dec << ", " << ms << " ms)\n";
        } else {
             std::cout << "FAIL (Found: 0x" << std::hex << h << ")\n";
             return 1; 
        }
    }

    std::cout << "\n  [PASS] All Logic Backends Verified Deterministically.\n";
    return 0;
}
