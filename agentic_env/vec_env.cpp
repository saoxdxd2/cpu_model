#include "agentic_env/vec_env.hpp"
#include <stdexcept>
#include <xmmintrin.h> // Intel Intrinsics for _mm_prefetch
#if defined(_MSC_VER)
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace nca::env {

void* alloc_aligned(size_t size, size_t alignment) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
    return ptr;
#endif
}

void free_aligned(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

VecEnv::VecEnv(std::vector<std::unique_ptr<Environment>> envs) 
    : envs_(std::move(envs)) {
    if (envs_.empty()) {
        throw std::invalid_argument("VecEnv requires >0 instances.");
    }
    
    obs_dim_ = envs_[0]->get_observation_dim();
    action_dim_ = envs_[0]->get_action_dim();
    
    for (const auto& e : envs_) {
        if (e->get_observation_dim() != obs_dim_ || e->get_action_dim() != action_dim_) {
            throw std::invalid_argument("Homogeneous dimension constraint violated.");
        }
    }

    size_t n = envs_.size();
    
    // Allocate contiguous SoA blocks. 64-byte boundary guarantees safe vmovaps usage.
    batched_obs_ = static_cast<float*>(alloc_aligned(n * obs_dim_ * sizeof(float), 64));
    rewards_ = static_cast<float*>(alloc_aligned(n * sizeof(float), 64));
    
    terminated_ = static_cast<bool*>(alloc_aligned(n * sizeof(bool), 64));
    truncated_ = static_cast<bool*>(alloc_aligned(n * sizeof(bool), 64));
}

VecEnv::~VecEnv() {
    free_aligned(batched_obs_);
    free_aligned(rewards_);
    free_aligned(terminated_);
    free_aligned(truncated_);
}

const float* VecEnv::reset_all() {
    for (size_t i = 0; i < envs_.size(); ++i) {
        envs_[i]->reset(batched_obs_ + i * obs_dim_);
    }
    return batched_obs_;
}

VecEnv::BatchedStepResult VecEnv::step(const float* batched_actions) {
    size_t n = envs_.size();
    
    for (size_t i = 0; i < n; ++i) {
        // Pipeline Optimization: Prefetch the NEXT environment's memory into L1d 
        // while the CPU is executing the current environment's step logic.
        if (i + 1 < n) {
            _mm_prefetch(reinterpret_cast<const char*>(batched_obs_ + (i + 1) * obs_dim_), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(batched_actions + (i + 1) * action_dim_), _MM_HINT_T0);
        }

        const float* action_in = batched_actions + i * action_dim_;
        float* obs_out = batched_obs_ + i * obs_dim_;
        
        StepResult res = envs_[i]->step(action_in, obs_out);
        
        // Auto-reset directly writes into the batched array, bypassing copy overhead.
        if (res.terminated || res.truncated) {
            envs_[i]->reset(obs_out);
        }
        
        rewards_[i] = res.reward;
        terminated_[i] = res.terminated;
        truncated_[i] = res.truncated;
    }
    
    return {batched_obs_, rewards_, terminated_, truncated_};
}

void VecEnv::get_action_masks(float* batched_masks_out) const {
    size_t offset = 0;
    for (size_t i = 0; i < envs_.size(); ++i) {
        envs_[i]->get_action_mask(batched_masks_out + offset);
        offset += action_dim_;
    }
}

void VecEnv::set_difficulty(int level) {
    for (auto& env : envs_) {
        env->set_difficulty(level);
    }
}

void VecEnv::set_reward_config(const RewardConfig& rc) {
    for (auto& env : envs_) {
        env->set_reward_config(rc);
    }
}

} // namespace nca::env
