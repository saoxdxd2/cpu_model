#pragma once
#include "agentic_env/env.hpp"
#include <cstdint>
#include <immintrin.h>

namespace nca::env {

constexpr size_t MAX_ENTITIES = 16;
constexpr size_t MAP_WIDTH    = 32;
constexpr size_t MAP_HEIGHT   = 16;
constexpr size_t MAP_CELLS    = MAP_WIDTH * MAP_HEIGHT; // 512

// 512-bit bitboard. Fits in exactly one AVX-512 register (__m512i)
// or two AVX2 registers (__m256i). Enables evaluating the entire board in 1-2 clock cycles.
struct Bitboard512 {
    uint64_t blocks[8];
    
    inline bool get(size_t idx) const { return (blocks[idx / 64] >> (idx % 64)) & 1ULL; }
    inline void set(size_t idx) { blocks[idx / 64] |= (1ULL << (idx % 64)); }
    inline void clear(size_t idx) { blocks[idx / 64] &= ~(1ULL << (idx % 64)); }
};

// Pure C++20 SoA layout, aligned to 64 bytes for cache-line perfection.
// The engine will operate on this struct directly using SIMD instructions.
struct alignas(64) GridState {
    Bitboard512 obstacles;      // 64 bytes
    Bitboard512 occupancy;      // 64 bytes
    Bitboard512 line_of_sight;  // 64 bytes
    
    // Entity SoA
    uint16_t entity_hp[MAX_ENTITIES];  // 32 bytes
    uint16_t entity_pos[MAX_ENTITIES]; // 32 bytes (Cell index [0, 511])
    uint8_t  entity_ap[MAX_ENTITIES];  // 16 bytes
    uint8_t  entity_mp[MAX_ENTITIES];  // 16 bytes
    uint8_t  entity_team[MAX_ENTITIES];// 16 bytes
    bool     entity_alive[MAX_ENTITIES]; // 16 bytes
};

// The AVX-Optimized Tactical Environment
class TacticalGridEnv : public Environment {
public:
    TacticalGridEnv();
    ~TacticalGridEnv() override = default;

    void reset(float* obs_out) override;
    StepResult step(const float* action_in, float* obs_out) override;
    
    size_t get_observation_dim() const override;
    size_t get_action_dim() const override;
    void get_action_mask(float* mask_out) const override;
    void set_difficulty(int level) override;
    void set_reward_config(const RewardConfig& rc) override { reward_cfg_ = rc; }

private:
    GridState state_;
    int tick_;
    int difficulty_level_;
    RewardConfig reward_cfg_;

    // Core logic loops
    void resolve_actions(const float* action_in);
    void update_line_of_sight();
    
    // Renders the compact 512-bit bitboards into the contiguous 
    // float tensor required by the MultimodalEngine.
    void render_observation(float* obs_out) const;
};

} // namespace nca::env
