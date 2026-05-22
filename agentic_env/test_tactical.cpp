#include "agentic_env/tactical_env.hpp"
#include "agentic_env/vec_env.hpp"
#include <iostream>
#include <chrono>
#include <cassert>
#include <vector>
#include <memory>

using namespace nca::env;

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << std::endl; g_fail++; } \
} while(0)

// ============================================================================
// TEST: TacticalEnv Initialization & Dimension
// ============================================================================
void test_tactical_init() {
    TacticalGridEnv env;
    CHECK(env.get_observation_dim() == 1616, "Obs dim should be 1616");
    CHECK(env.get_action_dim() == 80, "Action dim should be 80");
    
    float* obs = static_cast<float*>(alloc_aligned(1616 * sizeof(float), 64));
    env.reset(obs);
    
    // Check Bitboard Render (first 512 are obstacles)
    // Map cell 0 is generated as (0*13)%7 == 0 -> true
    CHECK(obs[0] == 1.0f, "Cell 0 should be an obstacle");
    
    // Map cell 1 is (1*13)%7 == 6 -> false
    CHECK(obs[1] == 0.0f, "Cell 1 should NOT be an obstacle");
    
    free_aligned(obs);
    std::cout << "[OK] TacticalGridEnv Init & Observation logic." << std::endl;
}

// ============================================================================
// TEST: TacticalEnv Step Execution
// ============================================================================
void test_tactical_step() {
    TacticalGridEnv env;
    float* obs = static_cast<float*>(alloc_aligned(1616 * sizeof(float), 64));
    float* act = static_cast<float*>(alloc_aligned(80 * sizeof(float), 64));
    
    env.reset(obs);
    
    for (int i = 0; i < 80; ++i) act[i] = 0.1f;
    act[1] = 0.9f; // Entity 0 takes action 1
    
    StepResult res = env.step(act, obs);
    CHECK(res.terminated == false, "Should not terminate on step 1");
    
    free_aligned(obs);
    free_aligned(act);
    std::cout << "[OK] TacticalGridEnv Step execution." << std::endl;
}

// ============================================================================
// PERF: AVX-Optimized Observation Rendering
// ============================================================================
void perf_render() {
    TacticalGridEnv env;
    float* obs = static_cast<float*>(alloc_aligned(1616 * sizeof(float), 64));
    env.reset(obs);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    const int ITERS = 1'000'000;
    
    for (int i = 0; i < ITERS; ++i) {
        // We directly call the rendering by resetting
        env.reset(obs);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
    
    std::cout << "[PERF] TacticalGridEnv reset+render: " << (ms * 1e6) / ITERS 
              << " ns/call (" << ITERS / (ms / 1000.0) << " fps)" << std::endl;
              
    free_aligned(obs);
}

// ============================================================================
// PERF: Batched Tactical Vector Environment
// ============================================================================
void perf_batched_tactical() {
    const size_t NUM_ENVS = 32;
    std::vector<std::unique_ptr<Environment>> envs;
    for(size_t i = 0; i < NUM_ENVS; ++i) envs.push_back(std::make_unique<TacticalGridEnv>());
    
    VecEnv vec_env(std::move(envs));
    vec_env.reset_all();
    
    float* acts = static_cast<float*>(alloc_aligned(NUM_ENVS * 80 * sizeof(float), 64));
    for (size_t i = 0; i < NUM_ENVS * 80; ++i) acts[i] = 0.5f;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    const int STEPS = 100'000; // 3.2M total env steps
    
    for (int i = 0; i < STEPS; ++i) {
        vec_env.step(acts);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
    
    std::cout << "[PERF] VecEnv[32 x TacticalGridEnv]: " 
              << ((double)STEPS * NUM_ENVS) / (ms / 1000.0) << " env_steps/sec" << std::endl;
              
    free_aligned(acts);
}

int main() {
    std::cout << "=== TacticalGridEnv Validation ===" << std::endl;
    test_tactical_init();
    test_tactical_step();
    
    if (g_fail > 0) return 1;
    
    std::cout << "\n=== Tactical Performance ===" << std::endl;
    perf_render();
    perf_batched_tactical();
    
    return 0;
}
