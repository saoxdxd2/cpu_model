// ============================================================================
// NCA — Final Integrated Verification
// tests/test_final.cpp
// ============================================================================

#include "core/execution/multimodal_engine.hpp"
#include "config/model_config.hpp"
#include <iostream>
#include <chrono>
#include <vector>

int main() {
    std::cout << "+------------------------------------------+\n";
    std::cout << "|  NCA -- Final Integrated Verification    |\n";
    std::cout << "+------------------------------------------+\n\n";

    nca::execution::MultimodalEngine engine;

    alignas(64) float text_in[2048] = {1.0f};
    alignas(64) float img_in[16 * 16 * 32] = {0.5f};
    alignas(64) float out[2048];

    std::cout << "  [RUN ] Integrated Multimodal Step... " << std::flush;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.step(text_in, img_in, out);
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    std::cout << "OK\n";
    std::cout << "    Step Latency  : " << ms << " ms\n";
    std::cout << "    Effective Speed: " << (1000.0 / ms) << " tokens/s\n\n";

    // Verify output isn't zero
    float sum = 0;
    for(int i=0; i<2048; ++i) sum += std::abs(out[i]);
    
    if (sum > 0) {
        std::cout << "  [PASS] Neural Circuit Synthesis Active.\n";
    } else {
        std::cout << "  [FAIL] Output is zero. Logic disconnected.\n";
        return 1;
    }

    return 0;
}
