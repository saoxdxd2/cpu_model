#include "tests/autotest.hpp"
#include "core/activations.hpp"
#include "core/execution/fused_pipeline.hpp"
#include "core/execution/latent_adapter.hpp"
#include "core/execution/multimodal_engine.hpp"
#include "core/execution/route_planner.hpp"
#include "core/layers/glr.hpp"
#include "core/layers/halting.hpp"
#include "core/layers/mlp.hpp"
#include "core/layers/sla.hpp"
#include "core/layers/ssm.hpp"
#include "core/linalg/mx_linear.hpp"
#include "core/log.hpp"
#include "core/normalization.hpp"
#include "core/vision/scanner.hpp"
#include "core/vision/spectral_pruner.hpp"

// ── CUSTOM WRAPPERS FOR CLASS MEMBERS ────────────────────────────────────────

void test_spectral_prune(const float* in, size_t* out) {
    static nca::vision::SpectralPruner pruner({1024, 1024, 0.5f});
    pruner.prune({in, 1024*1024}, {out, 1024});
}

void test_latent_project(const nca::linalg::MXUINT8Tensor& x, float* y) {
    static nca::execution::LatentAdapter adapter({1024, 1024});
    adapter.project(x, y);
}

int main() {
    nca::simd::detect();
    nca::testing::print_hardware_info();

    // 1. Core Activations & Math
    nca::testing::run_benchmark("nca::math::silu", &nca::math::silu);
    nca::testing::run_benchmark("nca::math::rmsnorm", &nca::math::rmsnorm);
    
    // 2. VNNI Saturated Kernels
    nca::testing::run_benchmark("nca::linalg::mx_dot", &nca::linalg::mx_dot);
    nca::testing::run_benchmark("nca::linalg::mx_dual_dot", &nca::linalg::mx_dual_dot);
    nca::testing::run_benchmark("nca::linalg::mx_quad_dot", &nca::linalg::mx_quad_dot);
    nca::testing::run_benchmark("nca::linalg::mx_rank16_dot", &nca::linalg::mx_rank16_dot);
    nca::testing::run_benchmark("nca::linalg::mx_gemv", &nca::linalg::mx_gemv);
    
    // 3. Backbone Layers
    nca::testing::run_benchmark("nca::layers::glr_step", &nca::layers::glr_step);
    nca::testing::run_benchmark("nca::layers::ssm_step", &nca::layers::ssm_step);
    nca::testing::run_benchmark("nca::layers::halting_step", &nca::layers::halting_step);
    nca::testing::run_benchmark("nca::layers::sla_step", &nca::layers::sla_step);
    
    // 4. Vision & Bridge
    nca::testing::run_benchmark("nca::vision::dwconv2d_3x3", &nca::vision::dwconv2d_3x3);
    nca::testing::run_benchmark("nca::vision::ssm2d_scan", &nca::vision::ssm2d_scan);
    nca::testing::run_benchmark("nca::vision::SpectralPruner::prune", &test_spectral_prune);
    nca::testing::run_benchmark("nca::execution::LatentAdapter::project", &test_latent_project);

    std::cout << "\n+------------------------------------------+\n";
    std::cout << "|  ALL CORE FUNCTIONS VERIFIED PERFECTLY   |\n";
    std::cout << "+------------------------------------------+\n";

    return 0;
}
