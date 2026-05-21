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

    // Initialize engine (uses Fixed Seed Contract)
    nca::execution::MultimodalEngine engine;

    const size_t D = nca::config::D_MODEL;
    alignas(64) float text_in[2048];
    alignas(64) float img_in[16 * 16 * 128];
    alignas(64) float out[2048];

    // 1. Initialize with distinct stable signal
    for(int i=0; i<D; ++i) text_in[i] = (float)i / D;
    for(int i=0; i<16*16*128; ++i) img_in[i] = (float)(i % 100) / 100.0f;

    std::cout << "  [RUN ] Integrated Multimodal Step... " << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.step(text_in, img_in, out);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "OK (" << ms << " ms)\n";

    // 2. GOLDEN REFERENCE VALIDATION
    uint64_t output_hash = fnv1a_hash(out, D);
    
    // Stable Golden Hash for v6.1 (Ice Lake / MSVC)
    const uint64_t GOLDEN_HASH = 0xcd9881817ff8b5ULL; 

    std::cout << "  [CHECK] Golden Parameter Contract... ";
    
    if (output_hash == GOLDEN_HASH) {
        std::cout << "OK (Hash: 0x" << std::hex << output_hash << std::dec << ")\n";
        std::cout << "  [PASS] Neural Circuit Synthesis validated via Deterministic Contract.\n";
    } else {
        std::cout << "FAIL\n";
        std::cout << "  [ERR ] Hash mismatch (Found 0x" << std::hex << output_hash << "). Logic changed.\n";
        return 1;
    }

    std::cout << "  [INFO] Effective Performance: " << (1000.0 / ms) << " tokens/s\n\n";

    return 0;
}
