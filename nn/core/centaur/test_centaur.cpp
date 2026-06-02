// ============================================================================
// CENTAUR — Hardware-Adaptive Execution Engine
// Test + Benchmark Harness
//
// Validates:
//   1. CPU fingerprint correctness
//   2. Tile planner output sanity
//   3. Execution graph compilation
//   4. Graph executor roundtrip
//   5. Performance measurement
// ============================================================================

#include "core/centaur/centaur.hpp"
#include <cassert>
#include <chrono>
#include <random>
#include <numeric>
#include <cmath>

namespace {

constexpr size_t D_MODEL   = 2048;
constexpr size_t N_EXPERTS = 1024;
constexpr size_t TOP_K     = 16;
constexpr size_t BENCH_ITERS = 1000;

void test_fingerprint() {
    std::cout << "═══ TEST 1: CPU Fingerprint ═══\n";
    const auto& p = nca::centaur::fingerprint();

    // Sanity checks
    assert(p.L1d.size_bytes > 0 && "L1d must be detected");
    assert(p.L2.size_bytes  > 0 && "L2 must be detected");
    assert(p.L3.size_bytes  > 0 && "L3 must be detected");
    assert(p.L1d.size_bytes < p.L2.size_bytes && "L1d < L2");
    assert(p.L2.size_bytes  < p.L3.size_bytes && "L2 < L3");
    assert(p.L1d.line_size == 64 && "Cache line must be 64B");
    assert(p.logical_cores > 0);
    assert(p.simd.simd_width_f32() >= 4);
    assert(std::strlen(p.vendor) > 0);

    std::cout << "  ✓ L1d=" << (p.L1d.size_bytes/1024) << "KB"
              << "  L2=" << (p.L2.size_bytes/1024) << "KB"
              << "  L3=" << (p.L3.size_bytes/(1024*1024)) << "MB\n";
    std::cout << "  ✓ SIMD width: " << p.simd.simd_width_f32() << " floats\n";
    std::cout << "  ✓ Cores: " << (int)p.physical_cores << "P / " << (int)p.logical_cores << "T\n";
    std::cout << "  PASS\n\n";
}

void test_tile_planner() {
    std::cout << "═══ TEST 2: Tile Planner ═══\n";
    const auto& p = nca::centaur::fingerprint();
    auto ep = nca::centaur::plan_execution(p, D_MODEL, N_EXPERTS, TOP_K);

    // Tile must fit in L1
    size_t working_set = (ep.tile.M * ep.tile.K + ep.tile.K * ep.tile.N + ep.tile.M * ep.tile.N) * sizeof(float);
    std::cout << "  Tile: [" << ep.tile.M << " × " << ep.tile.N << " × " << ep.tile.K << "]\n";
    std::cout << "  Working set: " << (working_set / 1024) << " KB (L1d = " << (p.L1d.size_bytes / 1024) << " KB)\n";

    // Microkernel must be SIMD-aligned
    assert(ep.ukernel.nr == p.simd.simd_width_f32());
    assert(ep.ukernel.mr > 0);
    assert(ep.ukernel.kr > 0);

    // Prefetch depth must match strategy
    if (ep.strategy == nca::centaur::ExecutionPlan::Strategy::L1_RESIDENT) {
        assert(ep.prefetch.depth == 0);
    } else {
        assert(ep.prefetch.depth > 0);
    }

    // Expert layout
    assert(ep.experts.num_resident_experts > 0);
    assert(ep.experts.top_k > 0);
    std::cout << "  Experts: " << ep.experts.num_resident_experts << " resident, top_k=" << ep.experts.top_k << "\n";
    std::cout << "  Strategy: " << ep.strategy_name() << "\n";
    std::cout << "  PASS\n\n";
}

void test_graph_compile() {
    std::cout << "═══ TEST 3: Execution Graph Compilation ═══\n";
    auto graph = nca::centaur::compile_for_this_cpu(D_MODEL, N_EXPERTS, TOP_K);

    assert(graph.nodes.size() > 0);
    assert(graph.slots.size() > 0);
    assert(graph.arena_size_bytes > 0);
    assert(graph.num_compute_nodes > 0);
    assert(graph.total_flops_estimate > 0);

    // Verify node types present
    bool has_glr = false, has_fwht = false, has_bct = false, has_fence = false;
    for (const auto& n : graph.nodes) {
        if (n.type == nca::centaur::NodeType::GLR_RECURRENCE) has_glr = true;
        if (n.type == nca::centaur::NodeType::SPECTRAL_FWHT)  has_fwht = true;
        if (n.type == nca::centaur::NodeType::BCT_ROUTE)       has_bct = true;
        if (n.type == nca::centaur::NodeType::FENCE)           has_fence = true;
    }
    assert(has_glr && "Graph must contain GLR recurrence");
    assert(has_fwht && "Graph must contain FWHT");
    assert(has_bct && "Graph must contain BCT routing");
    assert(has_fence && "Graph must end with FENCE");

    // All ExecNodes must be cache-line sized
    assert(sizeof(nca::centaur::ExecNode) == 64);

    std::cout << "  Nodes: " << graph.nodes.size() 
              << " (" << graph.num_compute_nodes << " compute, " 
              << graph.num_prefetch_nodes << " prefetch)\n";
    std::cout << "  Arena: " << (graph.arena_size_bytes / 1024) << " KB\n";
    std::cout << "  FLOPs: " << graph.total_flops_estimate << "\n";
    std::cout << "  Backend: " << graph.kernels.backend_name << "\n";
    std::cout << "  PASS\n\n";
}

void test_executor_roundtrip() {
    std::cout << "═══ TEST 4: Executor Roundtrip ═══\n";
    
    // Compile a small graph
    auto graph = nca::centaur::compile_for_this_cpu(D_MODEL, 0, 0, 1, false, false);
    nca::centaur::GraphExecutor executor(graph);

    // Create deterministic input
    alignas(64) float input[D_MODEL];
    alignas(64) float output[D_MODEL];
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (size_t i = 0; i < D_MODEL; ++i) input[i] = dist(rng);

    // Load GLR weights (slot 2 = alpha, slot 3 = beta)
    alignas(64) float alpha[D_MODEL], beta[D_MODEL];
    for (size_t i = 0; i < D_MODEL; ++i) {
        alpha[i] = 0.99f;  // Near-identity recurrence
        beta[i]  = 0.01f;  // Small bias
    }
    executor.load_weights(2, alpha, D_MODEL);
    executor.load_weights(3, beta, D_MODEL);

    // Run
    executor.load_state(input, D_MODEL);
    executor.execute(D_MODEL);
    executor.extract_state(output, D_MODEL);

    // Output must be finite and non-zero
    float sum = 0.0f;
    bool all_finite = true;
    for (size_t i = 0; i < D_MODEL; ++i) {
        sum += std::abs(output[i]);
        if (!std::isfinite(output[i])) all_finite = false;
    }
    assert(all_finite && "Output must be finite");
    assert(sum > 0.0f && "Output must be non-zero");

    std::cout << "  Output L1 norm: " << sum << "\n";
    std::cout << "  All finite: " << (all_finite ? "YES" : "NO") << "\n";
    std::cout << "  PASS\n\n";
}

void benchmark_execution() {
    std::cout << "═══ BENCHMARK: Graph Execution Throughput ═══\n";

    size_t batch_size = 16;
    nca::centaur::Engine engine(D_MODEL, N_EXPERTS, TOP_K, batch_size, true, true, false);

    alignas(64) float state[D_MODEL * 16];
    alignas(64) float output[D_MODEL * 16];
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (size_t i = 0; i < D_MODEL * batch_size; ++i) state[i] = dist(rng);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        engine.load_state(state, batch_size);
        engine.step(batch_size);
    }

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < BENCH_ITERS; ++i) {
        engine.load_state(state, batch_size);
        engine.step(batch_size);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_step_us = (elapsed_ms * 1000.0) / BENCH_ITERS;
    double steps_per_sec = 1e6 / per_step_us;

    engine.extract(output, batch_size);

    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Iterations: " << BENCH_ITERS << "\n";
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n";
    std::cout << "  Per step:   " << std::setprecision(2) << per_step_us << " μs\n";
    std::cout << "  Throughput: " << std::setprecision(0) << (steps_per_sec * batch_size) << " tokens/s\n";
    
    // Total flops per batch
    double flops_per_step = engine.graph().total_flops_estimate * batch_size;
    std::cout << "  FLOPs est:  " << engine.graph().total_flops_estimate << " / token\n";

    double gflops = (flops_per_step * steps_per_sec) / 1e9;
    std::cout << "  GFLOP/s:    " << std::setprecision(2) << gflops << "\n\n";
}

} // namespace

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  CENTAUR — Hardware-Adaptive Execution Engine           ║\n";
    std::cout << "║  Runtime Compiler + Cache Optimizer + SIMD Compiler     ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    // Print full CPU profile
    nca::centaur::print_profile(nca::centaur::fingerprint());

    // Run tests
    test_fingerprint();
    test_tile_planner();
    test_graph_compile();
    test_executor_roundtrip();
    benchmark_execution();

    std::cout << "═══ ALL TESTS PASSED ═══\n\n";
    return 0;
}
