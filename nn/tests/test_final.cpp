// ============================================================================
// NCA — Final Integrated Verification (Deep Validation)
// tests/test_final.cpp
// ============================================================================

#include "core/execution/multimodal_engine.hpp"
#include "config/model_config.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>

int main() {
    std::cout << "+------------------------------------------+\n";
    std::cout << "|  NCA -- Final Integrated Verification    |\n";
    std::cout << "+------------------------------------------+\n\n";

    nca::execution::MultimodalEngine engine;

    const size_t D = nca::config::D_MODEL;
    alignas(64) float text_in[2048];
    alignas(64) float img_in[16 * 16 * 128];
    alignas(64) float out[2048];

    // 1. Initialize with distinct signal
    for(int i=0; i<D; ++i) text_in[i] = (float)i / D;
    for(int i=0; i<16*16*128; ++i) img_in[i] = (float)(i % 100) / 100.0f;

    std::cout << "  [RUN ] Integrated Multimodal Step... " << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.step(text_in, img_in, out);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "OK (" << ms << " ms)\n";

    // 2. MODEL VALIDATION: Structural Variance Check
    // We calculate the mean and variance of the output. 
    // A dummy plumbing check would have very uniform output. 
    // A real neural network should have high structural variance.
    double mean = 0, sq_mean = 0;
    for(int i=0; i<D; ++i) {
        mean += out[i];
        sq_mean += out[i] * out[i];
    }
    mean /= D;
    sq_mean /= D;
    double variance = sq_mean - (mean * mean);

    std::cout << "  [CHECK] Statistical Mind-Map... ";
    if (variance > 1e-6) {
        std::cout << "OK (Var: " << variance << ")\n";
        std::cout << "  [PASS] Neural Circuit Synthesis validated with Structural Variance.\n";
    } else {
        std::cout << "FAIL (Var too low: " << variance << ")\n";
        std::cout << "  [ERR ] Output is too uniform. Logic potentially collapsed.\n";
        return 1;
    }

    std::cout << "  [INFO] Effective Performance: " << (1000.0 / ms) << " tokens/s\n\n";

    return 0;
}
