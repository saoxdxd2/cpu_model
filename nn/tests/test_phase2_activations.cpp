// ============================================================================
// NCA — Phase 2: AMI Activations Test
// tests/test_phase2_activations.cpp
//
// Validates the PRODUCTION API (core/activations.hpp) 
// using the AMI standard against PyTorch references.
// ============================================================================

#include "core/activations.hpp"
#include "core/simd/dispatch.hpp"

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  NCA — Phase 2: Activations API Test     ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    constexpr size_t D = 8192; 
    
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor x = torch::randn({D}, opts);

    // Reference LibTorch SiLU
    torch::Tensor ref_out = torch::silu(x);

    // Clone for our in-place test
    torch::Tensor out_tensor = x.clone();
    std::span<float> out_span(out_tensor.data_ptr<float>(), D);

    std::cout << "[1] Correctness Verification (Production API)\n";

    // Call official shipped API
    nca::math::silu(out_span);
    
    auto backend = nca::simd::best_backend();
    std::cout << "    Dispatched Backend: " << nca::simd::backend_name(backend) << "\n";

    const float* ref_ptr = ref_out.data_ptr<float>();
    const float* out_ptr = out_tensor.data_ptr<float>();

    float max_err = 0;
    for (size_t i = 0; i < D; ++i) {
        max_err = std::max(max_err, std::abs(ref_ptr[i] - out_ptr[i]));
    }

    std::cout << "    Max error         : " << max_err << "\n";

    if (max_err > 1e-3f) {
        std::cerr << "\n❌ Precision check failed! Error is too high.\n";
        return 1;
    }
    std::cout << "    ✓ API matches LibTorch within precision limits.\n\n";

    // 2. Benchmarking
    std::cout << "[2] Micro-Benchmark (10,000 iterations of D=8192)\n";
    constexpr int ITERS = 10000;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        nca::math::silu(out_span);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    std::cout << "    Production API    : " << ms << " ms total (" << (ms / ITERS) << " ms per iteration)\n";

    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "  Phase 2: Activations API PASS ✓\n";
    std::cout << "══════════════════════════════════════════\n";

    return 0;
}
