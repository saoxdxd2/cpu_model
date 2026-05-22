#include "agentic_env/tactical_env.hpp"
#include <cstring>
#include <immintrin.h>
#include <algorithm>

namespace nca::env {

TacticalGridEnv::TacticalGridEnv() : tick_(0), difficulty_level_(1) {
    std::memset(&state_, 0, sizeof(GridState));
}

size_t TacticalGridEnv::get_observation_dim() const {
    // 3 bitboards (512 each) + 16 entities * 5 spatial stats
    return (3 * 512) + (MAX_ENTITIES * 5); // 1616 floats
}

size_t TacticalGridEnv::get_action_dim() const {
    // 5 categorical choices per entity (Up, Down, Left, Right, Skip)
    return MAX_ENTITIES * 5; 
}

void TacticalGridEnv::set_difficulty(int level) {
    difficulty_level_ = level;
}

void TacticalGridEnv::reset(float* obs_out) {
    tick_ = 0;
    std::memset(&state_, 0, sizeof(GridState));
    
    // Deterministic pseudo-random terrain generation
    // Higher difficulty = denser obstacles
    size_t obstacle_mod = std::max(2, 7 - difficulty_level_);
    for (size_t i = 0; i < MAP_CELLS; ++i) {
        if ((i * 13) % obstacle_mod == 0) state_.obstacles.set(i);
    }
    
    // Spawn 8 vs 8 teams
    for (size_t i = 0; i < MAX_ENTITIES; ++i) {
        state_.entity_alive[i] = true;
        state_.entity_hp[i] = 100;
        state_.entity_ap[i] = 2;
        state_.entity_mp[i] = 4;
        state_.entity_team[i] = (i < MAX_ENTITIES / 2) ? 0 : 1;
        
        size_t pos = (i * 31) % MAP_CELLS;
        while(state_.obstacles.get(pos) || state_.occupancy.get(pos)) {
            pos = (pos + 1) % MAP_CELLS;
        }
        
        state_.entity_pos[i] = static_cast<uint16_t>(pos);
        state_.occupancy.set(pos);
    }
    
    update_line_of_sight();
    render_observation(obs_out);
}

StepResult TacticalGridEnv::step(const float* action_in, float* obs_out) {
    tick_++;
    resolve_actions(action_in);
    update_line_of_sight();
    render_observation(obs_out);
    
    // Reward: Configurable metrics
    float reward = reward_cfg_.step_penalty;
    int t0_alive = 0, t1_alive = 0;
    for (size_t i = 0; i < MAX_ENTITIES; ++i) {
        if (state_.entity_alive[i]) {
            if (state_.entity_team[i] == 0) t0_alive++;
            else t1_alive++;
        }
    }
    
    reward += static_cast<float>(t0_alive) * reward_cfg_.survival_bonus;
    reward -= static_cast<float>(t1_alive) * reward_cfg_.survival_bonus; // Kill translates to survival_bonus loss for enemies
    
    bool done = (tick_ >= 100) || (t0_alive == 0) || (t1_alive == 0);
    if (done && t0_alive > 0 && t1_alive == 0) {
        reward += reward_cfg_.win_bonus;
    }
    
    return {reward, done, false};
}

void TacticalGridEnv::resolve_actions(const float* action_in) {
    // Action tensor: [MAX_ENTITIES, 5]
    for (size_t i = 0; i < MAX_ENTITIES; ++i) {
        if (!state_.entity_alive[i]) continue;
        
        // Argmax interpretation of the continuous action logit
        int best_a = 0;
        float best_val = action_in[i * 5];
        for (int a = 1; a < 5; ++a) {
            if (action_in[i * 5 + a] > best_val) {
                best_val = action_in[i * 5 + a];
                best_a = a;
            }
        }
        
        // Dummy execution logic — simply deduct MP
        if (best_a > 0 && state_.entity_mp[i] > 0) {
            state_.entity_mp[i]--;
        }
    }
}

void TacticalGridEnv::update_line_of_sight() {
    // ── AVX2 BITBOARD MATH ──────────────────────────────────────────────
    // A complex spatial raycast is replaced by parallel bitwise operations.
    // Example: LoS is defined mathematically across the entire grid simultaneously.
    
#if defined(__AVX2__) || defined(__AVX__)
    // Load the 512-bit occupancy and obstacles directly into AVX registers
    __m256i occ0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state_.occupancy.blocks));
    __m256i occ1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state_.occupancy.blocks + 4));
    
    __m256i obs0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state_.obstacles.blocks));
    __m256i obs1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state_.obstacles.blocks + 4));
    
    // Mathematical logic: visible = occupancy ANDNOT obstacles 
    // (In reality, we'd add bit-shifts for spatial spreading here)
    __m256i los0 = _mm256_andnot_si256(obs0, occ0);
    __m256i los1 = _mm256_andnot_si256(obs1, occ1);
    
    // Store back to memory
    _mm256_store_si256(reinterpret_cast<__m256i*>(state_.line_of_sight.blocks), los0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(state_.line_of_sight.blocks + 4), los1);
#else
    for (int i = 0; i < 8; ++i) {
        state_.line_of_sight.blocks[i] = state_.occupancy.blocks[i] & ~state_.obstacles.blocks[i];
    }
#endif
}

void TacticalGridEnv::render_observation(float* obs_out) const {
    size_t out_idx = 0;
    
    // ── AVX2 BIT-TO-FLOAT EXPANSION (The "Smart Trick") ──────────────────
    // Extracts 512 bits into 512 floats without a single branch.
    // 1 bitboard (512 bits) = 64 loops processing 8 bits per loop.
#if defined(__AVX2__) || defined(__AVX__)
    const __m256i v_powers = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256  v_one    = _mm256_set1_ps(1.0f);
    
    auto render_bitboard = [&](const Bitboard512& bb) {
        for (int i = 0; i < 8; ++i) {
            uint64_t block = bb.blocks[i];
            for (int b = 0; b < 8; ++b) {
                uint8_t byte_val = static_cast<uint8_t>((block >> (b * 8)) & 0xFF);
                
                // Broadcast the byte to 8x 32-bit integers
                __m256i v_byte = _mm256_set1_epi32(byte_val);
                
                // Mask with powers of 2
                __m256i v_and = _mm256_and_si256(v_byte, v_powers);
                
                // Compare: Sets to 0xFFFFFFFF if bit is present, 0x0 otherwise
                __m256i v_mask = _mm256_cmpeq_epi32(v_and, v_powers);
                
                // Cast mask to float and bitwise AND with 1.0f
                __m256 v_out = _mm256_and_ps(_mm256_castsi256_ps(v_mask), v_one);
                
                _mm256_storeu_ps(obs_out + out_idx, v_out);
                out_idx += 8;
            }
        }
    };
#else
    auto render_bitboard = [&](const Bitboard512& bb) {
        for (int i = 0; i < 8; ++i) {
            uint64_t block = bb.blocks[i];
            for (int bit = 0; bit < 64; ++bit) {
                obs_out[out_idx++] = (block & (1ULL << bit)) ? 1.0f : 0.0f;
            }
        }
    };
#endif

    render_bitboard(state_.obstacles);
    render_bitboard(state_.occupancy);
    render_bitboard(state_.line_of_sight);
    
    for (size_t i = 0; i < MAX_ENTITIES; ++i) {
        obs_out[out_idx++] = state_.entity_alive[i] ? static_cast<float>(state_.entity_hp[i]) : 0.0f;
        obs_out[out_idx++] = static_cast<float>(state_.entity_pos[i] % MAP_WIDTH); // X
        obs_out[out_idx++] = static_cast<float>(state_.entity_pos[i] / MAP_WIDTH); // Y
        obs_out[out_idx++] = static_cast<float>(state_.entity_ap[i]);
        obs_out[out_idx++] = static_cast<float>(state_.entity_mp[i]);
    }
}

void TacticalGridEnv::get_action_mask(float* mask_out) const {
    const float INVALID = -1e9f;
    const float VALID = 0.0f;
    
    size_t idx = 0;
    for (size_t i = 0; i < MAX_ENTITIES; ++i) {
        // Condition: Entity must be alive to take ANY action
        float alive_mask = state_.entity_alive[i] ? VALID : INVALID;
        
        // Action 0: Skip turn (always valid if alive)
        mask_out[idx++] = alive_mask;
        
        // Actions 1-4: Move/Attack (requires AP/MP)
        float ap_mp_mask = (state_.entity_ap[i] > 0 || state_.entity_mp[i] > 0) ? VALID : INVALID;
        
        // Branchless combination: if dead (-1e9), ap_mp_mask also becomes heavily negative.
        // If alive (0), it relies on AP/MP check.
        float move_attack_mask = alive_mask + ap_mp_mask;
        
        mask_out[idx++] = move_attack_mask; // Action 1
        mask_out[idx++] = move_attack_mask; // Action 2
        mask_out[idx++] = move_attack_mask; // Action 3
        mask_out[idx++] = move_attack_mask; // Action 4
    }
}

} // namespace nca::env
