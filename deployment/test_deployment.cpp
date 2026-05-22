#include "deployment/nca_deploy.h"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout << "===========================================\n";
    std::cout << " NCA Deployment C-API Verification Test   \n";
    std::cout << "===========================================\n\n";

    const size_t OBS_DIM = 1616;
    const size_t ACT_DIM = 80;

    std::cout << "[1/3] Creating Silicon Engine via C-API...\n";
    nca_engine_ptr engine = nca_engine_create(OBS_DIM, ACT_DIM);
    if (!engine) {
        std::cerr << "  [FAIL] Failed to create engine.\n";
        return 1;
    }

    // 2. Simulate high-speed inference
    std::cout << "[2/3] Executing Latency Benchmark (1000 steps)...\n";
    std::vector<float> obs(OBS_DIM, 0.5f);
    std::vector<float> actions(ACT_DIM + 1);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        nca_engine_step(engine, obs.data(), actions.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  [OK] Average Latency: " << (total_ms / 1000.0) << " ms / step\n";

    std::cout << "[3/3] Destroying Engine...\n";
    nca_engine_destroy(engine);

    std::cout << "\n[SUCCESS] Deployment Bindings Verified.\n";
    return 0;
}
