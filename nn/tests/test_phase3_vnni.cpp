// ============================================================================
// NCA — Phase 3: MXINT8 VNNI Math Core Test
// tests/test_phase3_vnni.cpp
//
// Validates the PRODUCTION API (core/linalg/mx_linear.hpp)
// ============================================================================

#include "core/linalg/mx_linear.hpp"
#include <torch/torch.h>
#include <iostream>
#include <chrono>

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  NCA — Phase 3: MX VNNI API Test         ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    constexpr size_t D = 8192; 
    constexpr size_t BLOCKS = D / 32;
    
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor W_ref = torch::randn({D}, opts);
    torch::Tensor X_ref = torch::relu(torch::randn({D}, opts));

    float py_dot = torch::dot(W_ref, X_ref).item<float>();

    // Dynamic MX Quantization using pure C++ RAII API
    nca::linalg::MXINT8Tensor W_q(BLOCKS);
    nca::linalg::MXUINT8Tensor X_q(BLOCKS);
    
    nca::linalg::mx_quantize_w(W_ref.data_ptr<float>(), W_q);
    nca::linalg::mx_quantize_x(X_ref.data_ptr<float>(), X_q);

    std::cout << "[1] Correctness Verification\n";
    
    float vnni_dot = nca::linalg::mx_dot(W_q, X_q);
    float err = std::abs(py_dot - vnni_dot) / std::abs(py_dot);

    std::cout << "    LibTorch FP32 : " << py_dot << "\n";
    std::cout << "    VNNI MXINT8   : " << vnni_dot << "\n";
    std::cout << "    Relative Error: " << (err * 100.0f) << "%\n";

    if (err > 0.05f) { 
        std::cerr << "\n❌ Quantization loss is too high (>5%).\n";
        return 1;
    }
    std::cout << "    ✓ API perfectly preserves mathematical structure.\n\n";

    // 2. Benchmarking
    std::cout << "[2] Micro-Benchmark (10,000 iterations of D=8192)\n";
    constexpr int ITERS = 10000;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        volatile float d = nca::linalg::mx_dot(W_q, X_q);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_vnni = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "    MX VNNI Dot   : " << ms_vnni << " ms total (" << (ms_vnni/ITERS) << " ms/iter)\n";

    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "  Phase 3: VNNI API PASS ✓\n";
    std::cout << "══════════════════════════════════════════\n";

    return 0;
}
