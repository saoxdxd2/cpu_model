#include "agentic_env/sim_loop.hpp"
#include <immintrin.h>
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace nca::env {

SimLoop::SimLoop(SimLoopConfig cfg, 
                 std::unique_ptr<VecEnv> vec_env, 
                 std::shared_ptr<nca::execution::MultimodalEngine> engine)
    : cfg_(cfg), vec_env_(std::move(vec_env)), engine_(std::move(engine)) {
    
    if (!vec_env_ || !engine_) {
        throw std::invalid_argument("SimLoop requires valid VecEnv and MultimodalEngine");
    }

    size_t obs_dim = vec_env_->get_observation_dim();
    size_t act_dim = vec_env_->get_action_dim();

    // Capacity is precisely calculated to fit one full epoch.
    // Memory remains perfectly packed.
    size_t capacity = cfg_.rollout_steps * cfg_.num_envs;
    memory_ = std::make_unique<ReplayMemory>(capacity, obs_dim, act_dim);
    curriculum_ = std::make_unique<CurriculumScheduler>(vec_env_.get());

    // MultimodalEngine topology contract: [Actions..., Value]
    size_t engine_out_dim = act_dim + 1;

    batched_actions_ = static_cast<float*>(alloc_aligned(cfg_.num_envs * act_dim * sizeof(float), 64));
    batched_values_  = static_cast<float*>(alloc_aligned(cfg_.num_envs * sizeof(float), 64));
    batched_masks_   = static_cast<float*>(alloc_aligned(cfg_.num_envs * act_dim * sizeof(float), 64));
    engine_out_      = static_cast<float*>(alloc_aligned(cfg_.num_envs * engine_out_dim * sizeof(float), 64));

    // Spin up all parallel environments
    latest_obs_ = vec_env_->reset_all();
}

SimLoop::~SimLoop() {
    free_aligned(batched_actions_);
    free_aligned(batched_values_);
    free_aligned(batched_masks_);
    free_aligned(engine_out_);
}

void SimLoop::run_epoch() {
    size_t act_dim = vec_env_->get_action_dim();
    size_t obs_dim = vec_env_->get_observation_dim();
    size_t engine_out_dim = act_dim + 1;

    // ────────────────────────────────────────────────────────────────────────
    // PHASE 0: PREPARATION
    // ────────────────────────────────────────────────────────────────────────
    memory_->clear();

    // ────────────────────────────────────────────────────────────────────────
    // PHASE 1: HIGH-THROUGHPUT ROLLOUT
    // ────────────────────────────────────────────────────────────────────────
    for (size_t t = 0; t < cfg_.rollout_steps; ++t) {
        
        // 0. Query Laws of Physics (Action Masks)
        vec_env_->get_action_masks(batched_masks_);

        // 1. Neural Forward Pass (Geometric Schema)
        // We loop because AVX-512 lanes are now used for parallel logical tracking (Top-16), not batching
        for(size_t i=0; i<cfg_.num_envs; ++i) {
            engine_->step_geometric_env(latest_obs_ + i * obs_dim, engine_out_ + i * engine_out_dim, 0.2f);
        }

        // 2. Decode Logits and Apply Action Masks
        for (size_t i = 0; i < cfg_.num_envs; ++i) {
            float* e_out = engine_out_ + i * engine_out_dim;
            float* e_mask = batched_masks_ + i * act_dim;
            
            // Apply mask directly to logits (branchless addition)
            // e_mask is 0.0f (valid) or -1e9f (invalid)
#if defined(__AVX2__) || defined(__AVX__)
            for(size_t d = 0; d < act_dim; d += 8) {
                __m256 v_out = _mm256_loadu_ps(e_out + d);
                __m256 v_msk = _mm256_loadu_ps(e_mask + d);
                _mm256_storeu_ps(batched_actions_ + i * act_dim + d, _mm256_add_ps(v_out, v_msk));
            }
#else
            for (size_t d = 0; d < act_dim; ++d) {
                batched_actions_[i * act_dim + d] = e_out[d] + e_mask[d];
            }
#endif
            
            // Value is the final output neuron
            batched_values_[i] = e_out[act_dim];
        }

        // 3. Batched Environment Step
        VecEnv::BatchedStepResult res = vec_env_->step(batched_actions_);

        // [STABILITY] Convert masked logits to clean one-hot actions for memory
        // This prevents -1e9f from entering the training loss via t_actions.
        alignas(64) float clean_actions[32 * 80];
        std::memset(clean_actions, 0, sizeof(clean_actions));
        for (size_t i = 0; i < cfg_.num_envs; ++i) {
            for (size_t entity = 0; entity < 16; ++entity) {
                size_t base = i * act_dim + entity * 5;
                int best_a = 0;
                float best_v = batched_actions_[base];
                for (int a = 1; a < 5; ++a) {
                    if (batched_actions_[base + a] > best_v) {
                        best_v = batched_actions_[base + a];
                        best_a = a;
                    }
                }
                clean_actions[base + best_a] = 1.0f;
            }
        }

        // 4. Push to Compute Surface
        // 50% Bandwidth savings applied — next_states are NOT stored.
        memory_->push_batch(cfg_.num_envs, latest_obs_, clean_actions, res.rewards, res.terminated);
        memory_->write_values(cfg_.num_envs, batched_values_);

        latest_obs_ = res.next_observations;
    }

    // ────────────────────────────────────────────────────────────────────────
    // PHASE 2: ADVANTAGE ESTIMATION (L1-FUSED GAE)
    // ────────────────────────────────────────────────────────────────────────
    
    // Bootstrap the final transition V(s_T)
    for (size_t i = 0; i < cfg_.num_envs; ++i) {
        engine_->step_geometric_env(latest_obs_ + i * obs_dim, engine_out_ + i * engine_out_dim, 0.0f);
        batched_values_[i] = engine_out_[i * engine_out_dim + act_dim];
    }

    // Zero-Cost, Interleaved GAE sweep directly across L1 cache
    memory_->fused_compute_gae(cfg_.rollout_steps, cfg_.num_envs, cfg_.gamma, cfg_.lambda, batched_values_);

    // Optionally align distribution via Welford logic
    memory_->fused_normalize_observations(cfg_.rollout_steps * cfg_.num_envs);

    // ────────────────────────────────────────────────────────────────────────
    // PHASE 3: NEURAL UPDATE (PPO / RLS)
    // ────────────────────────────────────────────────────────────────────────
    // The ReplayMemory is now a contiguous, L1-hot memory mesh containing 
    // exactly what the engine needs: (S, A, R, GAE, V).
    
    BatchSpan span = memory_->get_latest_contiguous(cfg_.rollout_steps * cfg_.num_envs);
    const float* advantages = memory_->get_advantages();
    
    float mean_abs_adv = 0.0f;
    for (size_t i = 0; i < span.count; ++i) {
        mean_abs_adv += std::abs(advantages[i]);
    }

    // We feed the L1-hot memory mesh directly into the engine's Spectral RLS
    // [GEOMETRIC] The legacy VNNI update_from_trajectory is deprecated.
    // The Geometric Schema uses pointer-based DDQN which is slated for Phase 4.
    // engine_->update_from_trajectory(span.count, vec_env_->get_observation_dim(), act_dim, 
    //                                 span.states, span.actions, advantages);
    
    // Evaluate Curriculum Progression
    curriculum_->update_and_step_level(mean_abs_adv / span.count);
}

} // namespace nca::env
