#pragma once
#include "agentic_env/vec_env.hpp"
#include "agentic_env/curriculum_scheduler.hpp"
#include "agentic_env/replay_memory.hpp"
#include "core/execution/multimodal_engine.hpp"
#include <memory>

namespace nca::env {

struct SimLoopConfig {
    size_t num_envs = 32;
    size_t rollout_steps = 256;
    float gamma = 0.99f;
    float lambda = 0.95f;
    
    // RL Optimization Hyperparameters
    size_t ppo_epochs = 4;
    size_t mini_batch_size = 64;
};

// ============================================================================
// SIM LOOP ORCHESTRATOR
// ============================================================================
// The main heartbeat that synchronizes the high-throughput VecEnv with the 
// CPU-native MultimodalEngine. It acts as the bridge connecting the 
// tactical grid bitboards to the neural attention mechanisms.
class SimLoop {
public:
    SimLoop(SimLoopConfig cfg, 
            std::unique_ptr<VecEnv> vec_env, 
            std::shared_ptr<nca::execution::MultimodalEngine> engine);
    ~SimLoop();

    // Executes one full iteration of the Agentic Environment Loop:
    // 1. Data Collection (Rollout Phase)
    // 2. Advantage Estimation (L1-Fused GAE Phase)
    // 3. Neural Architecture Update (Learning Phase Placeholder)
    void run_epoch();
    
    // Provide access for the training pipeline
    VecEnv* get_vec_env() { return vec_env_.get(); }
    ReplayMemory* get_memory() { return memory_.get(); }

private:
    SimLoopConfig cfg_;
    std::unique_ptr<VecEnv> vec_env_;
    std::shared_ptr<nca::execution::MultimodalEngine> engine_;
    std::unique_ptr<ReplayMemory> memory_;
    std::unique_ptr<CurriculumScheduler> curriculum_;

    // Persistent observation buffer pointer from VecEnv
    const float* latest_obs_;

    // Hardware-aligned AVX buffers for zero-copy Engine I/O
    float* batched_actions_;
    float* batched_values_;
    float* batched_masks_;
    float* engine_out_;
};

} // namespace nca::env
