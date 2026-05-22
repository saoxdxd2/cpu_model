#pragma once
#include "agentic_env/env.hpp"
#include <vector>
#include <memory>

namespace nca::env {

class VecEnv {
public:
    // Takes ownership of environments. Allocates single SoA (Structure of Arrays) block for batching.
    explicit VecEnv(std::vector<std::unique_ptr<Environment>> envs);
    ~VecEnv();

    // Resets all envs. Writes to internal L1-aligned block and returns pointer to the zero-copy batch.
    const float* reset_all();

    struct BatchedStepResult {
        const float* next_observations; // 64-byte aligned [num_envs * obs_dim]
        const float* rewards;           // 64-byte aligned [num_envs]
        const bool* terminated;         // 64-byte aligned [num_envs]
        const bool* truncated;          // 64-byte aligned [num_envs]
    };

    // Fast-path execution across environments. 
    // Internally utilizes software prefetching (_mm_prefetch) to saturate L1d cache.
    BatchedStepResult step(const float* batched_actions);

    size_t num_envs() const { return envs_.size(); }
    size_t get_observation_dim() const { return obs_dim_; }
    size_t get_action_dim() const { return action_dim_; }

    // Queries action masks across all parallel environments
    void get_action_masks(float* batched_masks_out) const;

    // Curriculum Scheduler Hook
    void set_difficulty(int level);
    
    // Reward Config Hook
    void set_reward_config(const RewardConfig& rc);

private:
    std::vector<std::unique_ptr<Environment>> envs_;
    size_t obs_dim_;
    size_t action_dim_;

    // Contiguous SoA buffers pre-allocated and 64-byte aligned.
    // This allows the MultimodalEngine to read from batched_obs_ natively without pointer chasing.
    float* batched_obs_;
    float* rewards_;
    bool* terminated_;
    bool* truncated_;
};

} // namespace nca::env
