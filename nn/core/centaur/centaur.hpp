#pragma once
// ============================================================================
// CENTAUR — Hardware-Adaptive Execution Engine
// centaur/centaur.hpp
//
// THE ENTRY POINT.
//
// Usage:
//   auto& engine = nca::centaur::Engine::instance(d_model, n_experts, top_k);
//   engine.load_state(input);
//   engine.step();
//   engine.extract(output);
//
// What happens under the hood:
//   1. CPU fingerprinted (once)
//   2. Execution plan compiled from hardware constraints
//   3. Static DAG built with baked prefetch + pinned memory
//   4. Graph executed in a flat loop — zero dynamic decisions
//
// The model runs INSIDE the structure that hardware defines.
// ============================================================================

#include "core/centaur/cpu_fingerprint.hpp"
#include "core/centaur/tile_planner.hpp"
#include "core/centaur/execution_graph.hpp"
#include "core/centaur/graph_executor.hpp"
#include <iostream>
#include <iomanip>
#include <memory>
#include <chrono>

namespace nca::centaur {

// ── DIAGNOSTIC PRINTER ──────────────────────────────────────────────────────
inline void print_profile(const CPUProfile& p) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║        CENTAUR — CPU Fingerprint Report                 ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  CPU: " << std::left << std::setw(49) << p.brand << "║\n";
    std::cout << "║  Vendor: " << std::setw(46) << p.vendor << "║\n";
    std::cout << "║  Family: " << std::setw(4) << (int)p.family 
              << " Model: " << std::setw(4) << (int)p.model
              << " Stepping: " << std::setw(24) << (int)p.stepping << "║\n";
    std::cout << "║  Cores: " << (int)p.physical_cores << "P / " 
              << (int)p.logical_cores << "T" << std::setw(37) << "" << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  CACHE HIERARCHY                                        ║\n";
    std::cout << "║  L1d: " << std::setw(6) << (p.L1d.size_bytes / 1024) << " KB  "
              << std::setw(3) << p.L1d.associativity << "-way  "
              << std::setw(2) << p.L1d.latency_cycles << " cyc  "
              << std::setw(20) << (p.L1d.line_size == 64 ? "64B lines" : "???") << "║\n";
    std::cout << "║  L2:  " << std::setw(6) << (p.L2.size_bytes / 1024) << " KB  "
              << std::setw(3) << p.L2.associativity << "-way  "
              << std::setw(2) << p.L2.latency_cycles << " cyc" 
              << std::setw(24) << "" << "║\n";
    std::cout << "║  L3:  " << std::setw(6) << (p.L3.size_bytes / (1024*1024)) << " MB  "
              << std::setw(3) << p.L3.associativity << "-way  "
              << std::setw(2) << p.L3.latency_cycles << " cyc"
              << std::setw(24) << "" << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  SIMD CAPABILITIES                                      ║\n";
    std::cout << "║  AVX2:      " << (p.simd.avx2 ? "✓" : "✗")
              << "   AVX-512F:   " << (p.simd.avx512f ? "✓" : "✗")
              << "   VNNI: " << (p.simd.avx512vnni ? "✓" : "✗") 
              << std::setw(16) << "" << "║\n";
    std::cout << "║  BF16:      " << (p.simd.avx512bf16 ? "✓" : "✗")
              << "   AMX-TILE:   " << (p.simd.amx_tile ? "✓" : "✗")
              << "   AMX-INT8: " << (p.simd.amx_int8 ? "✓" : "✗")
              << std::setw(13) << "" << "║\n";
    std::cout << "║  SIMD Width: " << std::setw(3) << p.simd.simd_width_f32() 
              << " floats   Register File: " << std::setw(4) << p.simd.register_file_floats()
              << " floats" << std::setw(8) << "" << "║\n";
    std::cout << "║  Bandwidth: ~" << std::fixed << std::setprecision(1) << p.mem_bandwidth_gbps 
              << " GB/s (per-core est.)" << std::setw(21) << "" << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

inline void print_plan(const ExecutionPlan& ep) {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║        CENTAUR — Execution Plan                        ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Strategy: " << std::left << std::setw(44) << ep.strategy_name() << "║\n";
    std::cout << "║  Tile:     [" << ep.tile.M << " × " << ep.tile.N 
              << " × " << ep.tile.K << "]" << std::setw(30) << "" << "║\n";
    std::cout << "║  μKernel:  [" << ep.ukernel.mr << " × " << ep.ukernel.nr 
              << "]  kr=" << ep.ukernel.kr 
              << "  SIMD=" << ep.ukernel.simd_width << std::setw(20) << "" << "║\n";
    std::cout << "║  Prefetch: depth=" << ep.prefetch.depth 
              << "  stride=" << ep.prefetch.stride_bytes
              << "  NT=" << (ep.prefetch.use_nt_stores ? "Y" : "N")
              << std::setw(20) << "" << "║\n";
    if (ep.experts.num_resident_experts > 0) {
        std::cout << "║  Experts:  " << ep.experts.num_resident_experts << " resident"
                  << "  top_k=" << ep.experts.top_k 
                  << "  batch=" << ep.experts.routing_batch_size
                  << std::setw(14) << "" << "║\n";
    }
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";
}

inline void print_graph(const ExecutionGraph& g) {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║        CENTAUR — Compiled Execution Graph              ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Backend:   " << std::left << std::setw(43) << g.kernels.backend_name << "║\n";
    std::cout << "║  Nodes:     " << std::setw(6) << g.nodes.size()
              << " (" << g.num_compute_nodes << " compute + " << g.num_prefetch_nodes << " prefetch)"
              << std::setw(17) << "" << "║\n";
    std::cout << "║  Slots:     " << std::setw(6) << g.slots.size()
              << " (" << (g.arena_size_bytes / 1024) << " KB arena)"
              << std::setw(25) << "" << "║\n";
    std::cout << "║  Est FLOPs: " << std::setw(43) << g.total_flops_estimate << "║\n";

    // Node breakdown
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  DAG ORDER:                                             ║\n";
    
    const char* type_names[] = {
        "PREFETCH", "GEMM_TILE", "UNIFIED_BCT", "RMSNORM",
        "ACTIVATION", "RESIDUAL_ADD", "SPECTRAL_FWHT", "SPECTRAL_IFWHT",
        "GLR_RECURRENCE", "FENCE"
    };

    for (size_t i = 0; i < g.nodes.size(); ++i) {
        uint8_t t = static_cast<uint8_t>(g.nodes[i].type);
        const char* name = (t < 10) ? type_names[t] : "???";
        std::cout << "║  [" << std::setw(3) << i << "] " << std::setw(18) << name;
        
        if (g.nodes[i].type == NodeType::PREFETCH) {
            std::cout << " → slot " << g.nodes[i].prefetch_target_slot;
            std::cout << " (hint=" << (int)g.nodes[i].prefetch_hint << ")";
            std::cout << std::setw(17) << "";
        } else {
            std::cout << " in=" << g.nodes[i].input_slot 
                      << " out=" << g.nodes[i].output_slot;
            std::cout << std::setw(26) << "";
        }
        std::cout << "║\n";
    }

    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";
}

// ── ENGINE SINGLETON ────────────────────────────────────────────────────────
class Engine {
public:
    // Construct the CENTAUR engine for a given model geometry.
    // This performs: fingerprint → plan → compile → allocate arena.
    Engine(size_t d_model, size_t n_experts = 0, size_t top_k = 0,
           size_t max_batch_size = 32,
           bool use_spectral = true, bool use_bct = true, bool verbose = true)
        : d_model_(d_model)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        // ── 1. Fingerprint ──
        profile_ = fingerprint();

        // ── 2. Plan ──
        plan_ = plan_execution(profile_, d_model, n_experts, top_k);

        // ── 3. Compile ──
        GraphBuilder builder(profile_);
        graph_ = builder.compile(d_model, n_experts, top_k, max_batch_size, use_spectral, use_bct);

        // ── 4. Instantiate executor ──
        executor_ = std::make_unique<GraphExecutor>(graph_);

        auto t1 = std::chrono::high_resolution_clock::now();
        compile_time_us_ = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        if (verbose) {
            print_profile(profile_);
            print_plan(plan_);
            print_graph(graph_);
            std::cout << "CENTAUR compiled in " << compile_time_us_ << " μs\n\n";
        }
    }

    // ── Hot path ────────────────────────────────────────────────────────
    void load_state(const float* src, size_t batch_size = 1)  { executor_->load_state(src, d_model_, batch_size); }
    void step(size_t batch_size = 1)                          { executor_->execute(d_model_, batch_size); }
    void extract(float* dst, size_t batch_size = 1) const     { executor_->extract_state(dst, d_model_, batch_size); }
    
    void load_weights(uint16_t slot_id, const float* w, size_t count) {
        executor_->load_weights(slot_id, w, count);
    }

    // ── Introspection ───────────────────────────────────────────────────
    const CPUProfile&     profile()  const { return profile_; }
    const ExecutionPlan&  plan()     const { return plan_; }
    const ExecutionGraph& graph()    const { return graph_; }
    size_t                d_model()  const { return d_model_; }
    int64_t               compile_us() const { return compile_time_us_; }

private:
    size_t          d_model_;
    CPUProfile      profile_;
    ExecutionPlan   plan_;
    ExecutionGraph  graph_;
    std::unique_ptr<GraphExecutor> executor_;
    int64_t         compile_time_us_ = 0;
};

} // namespace nca::centaur
