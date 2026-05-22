#pragma once
#include <cstdint>
#include <cstddef>

namespace nca::env {

// Aligned memory allocators for AVX-512 (64-byte boundaries)
void* alloc_aligned(size_t size, size_t alignment = 64);
void free_aligned(void* ptr);

struct StepResult {
    float reward;
    bool terminated;
    bool truncated;
};

struct RewardConfig {
    float survival_bonus = 0.1f;
    float kill_bonus = 1.0f;
    float step_penalty = -0.01f;
    float win_bonus = 10.0f;
};

// Abstract Environment Interface (Zero-Copy)
// All buffers passed here must be 64-byte aligned to allow AVX-512 VMOVAPS instructions.
class Environment {
public:
    virtual ~Environment() = default;

    // Reset the environment and write the initial observation directly into `obs_out`.
    virtual void reset(float* obs_out) = 0;

    // Advance the environment by one step. 
    // Reads from `action_in`, writes the resulting state directly to `obs_out`.
    virtual StepResult step(const float* action_in, float* obs_out) = 0;

    // Static sizes required for ahead-of-time (AOT) allocation in VecEnv/ReplayMemory.
    virtual size_t get_observation_dim() const = 0;
    virtual size_t get_action_dim() const = 0;

    // Writes an action mask of size get_action_dim() into mask_out.
    // 0.0f = valid action, -1e9f = invalid action
    virtual void get_action_mask(float* mask_out) const = 0;

    // Curriculum Learning: Generalist difficulty adjustment
    virtual void set_difficulty(int level) {}
    
    // Configurable Reward System controlled by the Training Pipeline
    virtual void set_reward_config(const RewardConfig& rc) {}
};

} // namespace nca::env
