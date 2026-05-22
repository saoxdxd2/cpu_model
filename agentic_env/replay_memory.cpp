#include "agentic_env/replay_memory.hpp"
#include "agentic_env/env.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

namespace nca::env {

ReplayMemory::ReplayMemory(size_t capacity, size_t obs_dim, size_t action_dim) 
    : capacity_(capacity), size_(0), cursor_(0), 
      obs_dim_(obs_dim), action_dim_(action_dim), obs_count_(0) {
    
    tile_width_ = compute_tile_width(obs_dim_, action_dim_);

    states_      = static_cast<float*>(alloc_aligned(capacity_ * obs_dim_    * sizeof(float), 64));
    actions_     = static_cast<float*>(alloc_aligned(capacity_ * action_dim_ * sizeof(float), 64));
    rewards_     = static_cast<float*>(alloc_aligned(capacity_               * sizeof(float), 64));
    dones_       = static_cast<bool*> (alloc_aligned(capacity_               * sizeof(bool),  64));

    values_      = static_cast<float*>(alloc_aligned(capacity_ * sizeof(float), 64));
    advantages_  = static_cast<float*>(alloc_aligned(capacity_ * sizeof(float), 64));
    returns_     = static_cast<float*>(alloc_aligned(capacity_ * sizeof(float), 64));

    obs_mean_    = static_cast<float*>(alloc_aligned(obs_dim_ * sizeof(float), 64));
    obs_var_     = static_cast<float*>(alloc_aligned(obs_dim_ * sizeof(float), 64));
    
    clear();
}

ReplayMemory::~ReplayMemory() {
    free_aligned(states_);    free_aligned(actions_);
    free_aligned(rewards_);   free_aligned(dones_);
    free_aligned(values_);    free_aligned(advantages_);
    free_aligned(returns_);
    free_aligned(obs_mean_);  free_aligned(obs_var_);
}

void ReplayMemory::clear() {
    size_ = 0; cursor_ = 0; obs_count_ = 0;
    std::memset(obs_mean_, 0, obs_dim_ * sizeof(float));
    for (size_t i = 0; i < obs_dim_; ++i) obs_var_[i] = 1.0f;

    std::memset(values_,     0, capacity_ * sizeof(float));
    std::memset(advantages_, 0, capacity_ * sizeof(float));
    std::memset(returns_,    0, capacity_ * sizeof(float));
}

void ReplayMemory::push(const float* state, const float* action, float reward, bool done) {
    size_t idx = cursor_;
    std::memcpy(states_  + idx * obs_dim_,    state,  obs_dim_    * sizeof(float));
    std::memcpy(actions_ + idx * action_dim_, action, action_dim_ * sizeof(float));
    rewards_[idx] = reward;
    dones_[idx]   = done;
    cursor_ = (cursor_ + 1) % capacity_;
    if (size_ < capacity_) size_++;
}

void ReplayMemory::push_batch(size_t batch_size, const float* batched_states, const float* batched_actions, 
                              const float* rewards, const bool* dones) {
    size_t space = capacity_ - cursor_;
    if (batch_size <= space) {
        std::memcpy(states_  + cursor_ * obs_dim_,    batched_states,  batch_size * obs_dim_    * sizeof(float));
        std::memcpy(actions_ + cursor_ * action_dim_, batched_actions, batch_size * action_dim_ * sizeof(float));
        std::memcpy(rewards_ + cursor_,               rewards,         batch_size               * sizeof(float));
        std::memcpy(dones_   + cursor_,               dones,           batch_size               * sizeof(bool));
        cursor_ = (cursor_ + batch_size) % capacity_;
    } else {
        size_t first = space, second = batch_size - first;
        std::memcpy(states_  + cursor_ * obs_dim_,    batched_states,                        first * obs_dim_    * sizeof(float));
        std::memcpy(actions_ + cursor_ * action_dim_, batched_actions,                       first * action_dim_ * sizeof(float));
        std::memcpy(rewards_ + cursor_,               rewards,                               first               * sizeof(float));
        std::memcpy(dones_   + cursor_,               dones,                                 first               * sizeof(bool));
        
        std::memcpy(states_,  batched_states  + first * obs_dim_,    second * obs_dim_    * sizeof(float));
        std::memcpy(actions_, batched_actions + first * action_dim_, second * action_dim_ * sizeof(float));
        std::memcpy(rewards_, rewards         + first,               second               * sizeof(float));
        std::memcpy(dones_,   dones           + first,               second               * sizeof(bool));
        cursor_ = second % capacity_;
    }
    size_ = std::min(size_ + batch_size, capacity_);
}

size_t ReplayMemory::get_latest_spans(size_t count, BatchSpan* out_spans) const {
    if (size_ == 0 || count == 0) return 0;
    size_t actual = count < size_ ? count : size_;
    if (cursor_ >= actual) {
        size_t s = cursor_ - actual;
        out_spans[0] = {states_ + s*obs_dim_, actions_ + s*action_dim_, rewards_ + s, dones_ + s, actual};
        return 1;
    }
    size_t tc = actual - cursor_, ts = capacity_ - tc;
    out_spans[0] = {states_ + ts*obs_dim_, actions_ + ts*action_dim_, rewards_ + ts, dones_ + ts, tc};
    size_t n = 1;
    if (cursor_ > 0) {
        out_spans[1] = {states_, actions_, rewards_, dones_, cursor_};
        n = 2;
    }
    return n;
}

BatchSpan ReplayMemory::get_latest_contiguous(size_t count) const {
    if (size_ == 0 || count == 0) return {nullptr,nullptr,nullptr,nullptr,0};
    size_t actual = count < size_ ? count : size_;
    
    // If cursor is 0 and we are full, the latest 'actual' items are at the end of the buffer.
    if (cursor_ == 0 && size_ == capacity_) {
        size_t s = capacity_ - actual;
        return {states_ + s*obs_dim_, actions_ + s*action_dim_, rewards_ + s, dones_ + s, actual};
    }

    if (actual > cursor_) {
        // Not enough data before the cursor to satisfy 'actual' contiguously.
        // Return only what is available before the cursor.
        actual = cursor_;
    }

    if (actual == 0) return {nullptr,nullptr,nullptr,nullptr,0};
    
    size_t s = cursor_ - actual;
    return {states_ + s*obs_dim_, actions_ + s*action_dim_, rewards_ + s, dones_ + s, actual};
}

void ReplayMemory::write_values(size_t count, const float* values) {
    if (count == 0 || size_ == 0) return;
    size_t actual = count < size_ ? count : size_;
    if (cursor_ >= actual) {
        std::memcpy(values_ + (cursor_ - actual), values, actual * sizeof(float));
    } else {
        size_t tail = actual - cursor_;
        std::memcpy(values_ + (capacity_ - tail), values, tail * sizeof(float));
        std::memcpy(values_, values + tail, cursor_ * sizeof(float));
    }
}

void ReplayMemory::fused_normalize_observations(size_t count) {
    if (count == 0 || size_ == 0) return;
    size_t actual = count < size_ ? count : size_;
    
    size_t seg_starts[2], seg_counts[2], n_segs;
    if (cursor_ >= actual) {
        seg_starts[0] = cursor_ - actual; seg_counts[0] = actual; n_segs = 1;
    } else {
        seg_starts[0] = capacity_ - (actual - cursor_); seg_counts[0] = actual - cursor_;
        seg_starts[1] = 0; seg_counts[1] = cursor_; n_segs = 2;
    }

    const float eps = 1e-8f;

    for (size_t seg = 0; seg < n_segs; ++seg) {
        size_t base = seg_starts[seg];
        size_t n    = seg_counts[seg];

        for (size_t t = 0; t < n; ++t) {
            float* obs = states_ + (base + t) * obs_dim_;
            obs_count_++;
            float inv_n = 1.0f / static_cast<float>(obs_count_);
            size_t d = 0;

#if defined(__AVX2__) || defined(__AVX__)
            __m256 v_inv_n = _mm256_set1_ps(inv_n);
            __m256 v_eps   = _mm256_set1_ps(eps);

            for (; d + 8 <= obs_dim_; d += 8) {
                __m256 v_obs  = _mm256_load_ps(obs + d);
                __m256 v_mean = _mm256_load_ps(obs_mean_ + d);
                __m256 v_var  = _mm256_load_ps(obs_var_ + d);

                __m256 v_delta = _mm256_sub_ps(v_obs, v_mean);
                __m256 v_new_mean = _mm256_fmadd_ps(v_delta, v_inv_n, v_mean);
                __m256 v_delta2 = _mm256_sub_ps(v_obs, v_new_mean);
                
                __m256 v_dd = _mm256_mul_ps(v_delta, v_delta2);
                __m256 v_new_var = _mm256_fmadd_ps(_mm256_sub_ps(v_dd, v_var), v_inv_n, v_var);
                
                _mm256_store_ps(obs_mean_ + d, v_new_mean);
                _mm256_store_ps(obs_var_  + d, v_new_var);

                __m256 v_std = _mm256_sqrt_ps(_mm256_add_ps(v_new_var, v_eps));
                __m256 v_inv_std = _mm256_div_ps(_mm256_set1_ps(1.0f), v_std);
                __m256 v_norm = _mm256_mul_ps(v_delta2, v_inv_std);
                _mm256_store_ps(obs + d, v_norm);
            }
#endif
            for (; d < obs_dim_; ++d) {
                float delta    = obs[d] - obs_mean_[d];
                float new_mean = obs_mean_[d] + delta * inv_n;
                float delta2   = obs[d] - new_mean;
                float new_var  = obs_var_[d] + (delta * delta2 - obs_var_[d]) * inv_n;
                
                obs_mean_[d] = new_mean;
                obs_var_[d]  = new_var;
                obs[d] = delta2 / std::sqrt(new_var + eps);
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// L1-FUSED INTERLEAVED GAE
// ════════════════════════════════════════════════════════════════════════════
void ReplayMemory::fused_compute_gae(size_t rollout_steps, size_t num_envs, float gamma, float lambda, const float* final_next_values) {
    size_t total_items = rollout_steps * num_envs;
    if (total_items == 0 || size_ == 0) return;
    
    // We only compute GAE on the most recent `total_items`.
    // Ensure we don't exceed what's in the buffer.
    if (total_items > size_) {
        rollout_steps = size_ / num_envs;
        total_items = rollout_steps * num_envs;
    }
    
    size_t start_idx = (capacity_ + cursor_ - total_items) % capacity_;
    
    // Fast thread-local arrays for gae_acc and v_next to avoid heap allocation.
    // MAX_ENVS is practically capped by cache size anyway, 1024 is plenty.
    const size_t MAX_ENVS = 1024;
    float gae_acc[MAX_ENVS] = {0.0f};
    float v_next[MAX_ENVS];
    
    size_t active_envs = num_envs > MAX_ENVS ? MAX_ENVS : num_envs;
    std::memcpy(v_next, final_next_values, active_envs * sizeof(float));
    
    float gl = gamma * lambda;
    
    // Outer loop: Time (Backwards)
    for (int t = rollout_steps - 1; t >= 0; --t) {
        
        // Inner loop: Environments (Forward, strictly contiguous in memory!)
        for (size_t k = 0; k < active_envs; ++k) {
            size_t idx = (start_idx + t * num_envs + k);
            if (idx >= capacity_) idx -= capacity_; // Fast modulo
            
            float r_t    = rewards_[idx];
            float v_t    = values_[idx];
            float done_f = dones_[idx] ? 0.0f : 1.0f; 
            
            // δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            float delta = r_t + gamma * v_next[k] * done_f - v_t;
            
            // A_t = δ_t + γλ * A_{t+1} * (1 - done_t)
            gae_acc[k] = delta + gl * gae_acc[k] * done_f;
            
            advantages_[idx] = gae_acc[k];
            returns_[idx]    = gae_acc[k] + v_t;
            
            // V(s_{t+1}) for the next backward step is V(s_t) from this step
            v_next[k] = v_t;
        }
    }
}

} // namespace nca::env
