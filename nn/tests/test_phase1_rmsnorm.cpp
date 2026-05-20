// ============================================================================
// NCA — Phase 1: AMI RMSNorm Test
// tests/test_phase1_rmsnorm.cpp
//
// Atomic Mathematical Isolation (AMI) test for FP32 RMSNorm API.
// Validates results against LibTorch reference.
// Benchmark speed directly in C++.
// ============================================================================

#include "core/normalization.hpp"
#include "core/simd/dispatch.hpp"
#include "config/model_config.hpp"

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  NCA — Phase 1: RMSNorm API Test         ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    // 1. Setup Test Parameters
    constexpr int32_t D = nca::config::D_MODEL; // 2048
    constexpr float EPS = 1e-5f;
    
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor x = torch::randn({D}, opts);
    torch::Tensor weight = torch::ones({D}, opts); // Weights are 1 for easy testing

    // Reference LibTorch RMSNorm
    torch::Tensor x_var = (x * x).mean();
    torch::Tensor ref_out = x * torch::rsqrt(x_var + EPS) * weight;

    // Allocate output buffers
    std::vector<float> out_api(D);

    const float* in_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* ref_ptr = ref_out.data_ptr<float>();

    std::cout << "[1] Correctness Verification (API Dispatch)\n";

    // Run API implementation (which dynamically selects backend)
    nca::math::rmsnorm(out_api.data(), in_ptr, w_ptr, D, EPS);

    // Verify
    float max_err = 0;
    for (int i = 0; i < D; ++i) {
        max_err = std::max(max_err, std::abs(ref_ptr[i] - out_api[i]));
    }

    auto backend = nca::simd::best_backend();
    std::cout << "    Dispatched Backend: " << nca::simd::backend_name(backend) << "\n";
    std::cout << "    Max error         : " << max_err << "\n";
    
    if (max_err > 1e-4f) {
        std::cerr << "\n❌ Correctness check failed!\n";
        return 1;
    }
    std::cout << "    ✓ API matches LibTorch reference.\n\n";

    // 2. Benchmarking
    std::cout << "[2] Micro-Benchmark (100,000 iterations)\n";
    constexpr int ITERS = 100000;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        nca::math::rmsnorm(out_api.data(), in_ptr, w_ptr, D, EPS);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "    Time: " << ms << " ms\n";

    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "  Phase 1: RMSNorm API PASS ✓\n";
    std::cout << "══════════════════════════════════════════\n";

    return 0;
}
