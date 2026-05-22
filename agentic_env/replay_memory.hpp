#pragma once
#include <cstddef>
#include <cstdint>

namespace nca::env {

// ============================================================================
// L1 Cache Tiling Constants
// ============================================================================
constexpr size_t L1D_SIZE       = 32768;
constexpr size_t CACHE_LINE     = 64;

inline size_t compute_tile_width(size_t obs_dim, size_t action_dim) {
    // We eliminated next_states_.
    // Working set = state + action + reward + advantage
    size_t bytes_per_env = (obs_dim + action_dim + 2) * sizeof(float);
    size_t tile = L1D_SIZE / bytes_per_env;
    if (tile >= 8) return 8;
    if (tile >= 4) return 4;
    if (tile >= 2) return 2;
    return 1;
}

// Zero-copy view into contiguous SoA arrays
struct BatchSpan {
    const float* states;
    const float* actions;
    const float* rewards;
    const bool* dones;
    size_t count;
};

// ============================================================================
// L1-Fused Trajectory Buffer (Interleaved VecEnv Layout)
// ============================================================================
class ReplayMemory {
public:
    ReplayMemory(size_t capacity, size_t obs_dim, size_t action_dim);
    ~ReplayMemory();

    ReplayMemory(const ReplayMemory&) = delete;
    ReplayMemory& operator=(const ReplayMemory&) = delete;

    // ── RAW INGESTION (Removed next_state) ───────────────────────────────────
    void push(const float* state, const float* action, float reward, bool done);
    void push_batch(size_t batch_size, const float* batched_states, const float* batched_actions, 
                    const float* rewards, const bool* dones);

    // ── ZERO-COPY VIEWS ──────────────────────────────────────────────────────
    size_t get_latest_spans(size_t count, BatchSpan* out_spans) const;
    BatchSpan get_latest_contiguous(size_t count) const;

    // ── L1-FUSED COMPUTE KERNELS ─────────────────────────────────────────────

    // Normalize observations in-place
    void fused_normalize_observations(size_t count);

    // Write value function estimates for the latest `count` transitions.
    void write_values(size_t count, const float* values);

    // Computes GAE for interleaved VecEnv trajectories in a single backward sweep.
    // By processing inner-loop over K environments, memory access is 100% contiguous.
    // rollout_steps: T (number of timesteps per environment)
    // num_envs: K (number of parallel environments)
    // final_next_values: Array of size K containing V(s_T) for bootstrapping.
    void fused_compute_gae(size_t rollout_steps, size_t num_envs, float gamma, float lambda, const float* final_next_values);

    const float* get_advantages() const { return advantages_; }
    const float* get_values() const { return values_; }
    const float* get_returns() const { return returns_; }

    const float* get_obs_mean() const { return obs_mean_; }
    const float* get_obs_var()  const { return obs_var_;  }

    void clear();
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    size_t obs_dim() const { return obs_dim_; }
    size_t action_dim() const { return action_dim_; }
    size_t tile_width() const { return tile_width_; }

private:
    size_t capacity_;
    size_t size_;
    size_t cursor_;
    size_t obs_dim_;
    size_t action_dim_;
    size_t tile_width_;
    size_t obs_count_;

    // ── SoA TRAJECTORY STORAGE (50% Bandwidth Savings) ───────────────────────
    float* states_;        
    float* actions_;       
    float* rewards_;       
    bool*  dones_;         

    // ── COMPUTE SURFACES ─────────────────────────────────────────────────────
    float* values_;        
    float* advantages_;    
    float* returns_;       

    // ── RUNNING STATISTICS ───────────────────────────────────────────────────
    float* obs_mean_;      
    float* obs_var_;       
};

} // namespace nca::env
