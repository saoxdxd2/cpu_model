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

    uint64_t hash_a = 0, hash_b = 0, hash_c = 0;
    double ms_a = 0, ms_b = 0, ms_c = 0;

    // ── 1. VERIFY HASHED ROUTER (DEFAULT) ──────────────────────────────────
    {
        std::cout << "  [RUN ] Backend: HashedRouter (Default)... " << std::flush;
        nca::execution::MultimodalEngine engine;
        auto t0 = std::chrono::high_resolution_clock::now();
        engine.step(text_in, img_in, out);
        auto t1 = std::chrono::high_resolution_clock::now();
        ms_a = std::chrono::duration<double, std::milli>(t1 - t0).count();
        hash_a = fnv1a_hash(out, D);
        const uint64_t GOLDEN_H = 0x40612f1ed3b468ddULL;
        if (hash_a == GOLDEN_H) std::cout << "OK (Hash: 0x" << std::hex << hash_a << std::dec << ", " << ms_a << " ms)\n";
        else { std::cout << "FAIL (Found: 0x" << std::hex << hash_a << ")\n"; return 1; }
    }

    // ── 2. VERIFY SPECTRAL RLS (V7.2 - RIGOROUS) ─────────────────────────────
    {
        std::cout << "  [RUN ] Backend: SpectralRLS (Opt-in)... " << std::flush;
        nca::config::EngineConfig cfg;
        cfg.logic_backend = nca::config::LogicBackend::SpectralRLS;
        nca::execution::MultimodalEngine engine(cfg);
        auto t0 = std::chrono::high_resolution_clock::now();
        engine.step(text_in, img_in, out);
        auto t1 = std::chrono::high_resolution_clock::now();
        ms_b = std::chrono::duration<double, std::milli>(t1 - t0).count();
        hash_b = fnv1a_hash(out, D);
        const uint64_t GOLDEN_S = 0x1011119d947bbf5dULL; 
        if (hash_b == GOLDEN_S) std::cout << "OK (Hash: 0x" << std::hex << hash_b << std::dec << ", " << ms_b << " ms)\n";
        else { std::cout << "FAIL (Found: 0x" << std::hex << hash_b << ")\n"; return 1; }
    }

    // ── 3. VERIFY SPECTRAL ROUTED EXPERTS (V7.2 - RIGOROUS) ──────────────────
    {
        std::cout << "  [RUN ] Backend: SpectralRoutedExperts (Hybrid)... " << std::flush;
        nca::config::EngineConfig cfg;
        cfg.logic_backend = nca::config::LogicBackend::SpectralRoutedExperts;
        nca::execution::MultimodalEngine engine(cfg);
        auto t0 = std::chrono::high_resolution_clock::now();
        engine.step(text_in, img_in, out);
        auto t1 = std::chrono::high_resolution_clock::now();
        ms_c = std::chrono::duration<double, std::milli>(t1 - t0).count();
        hash_c = fnv1a_hash(out, D);
        const uint64_t GOLDEN_C = 0x67b7b772fa20193eULL; 
        if (hash_c == GOLDEN_C) std::cout << "OK (Hash: 0x" << std::hex << hash_c << std::dec << ", " << ms_c << " ms)\n";
        else { std::cout << "FAIL (Found: 0x" << std::hex << hash_c << ")\n"; return 1; }
    }

    // ── 4. COMPARISON & PERFORMANCE HARDENING ───────────────────────────────
    std::cout << "\n  [CHECK] Comparing Hash Uniqueness... ";
    if (hash_c != hash_a && hash_c != hash_b) {
        std::cout << "OK (Hybrid is unique)\n";
    } else {
        std::cout << "FAIL (Hybrid matches baseline)\n";
        return 1;
    }

    std::cout << "  [CHECK] Performance Latency Budget (<= 2x Default)... ";
    if (ms_c <= 2.0 * ms_a) {
        std::cout << "OK (" << ms_c << " ms <= 2x " << ms_a << " ms)\n";
    } else {
        std::cout << "FAIL (Exceeded Budget: " << ms_c << " ms > 2x " << ms_a << " ms)\n";
        return 1; 
    }

    std::cout << "\n  [PASS] All Logic Backends Verified and Performance-Hardened.\n";
    return 0;
}
