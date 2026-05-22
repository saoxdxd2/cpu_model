#pragma once
#include "agentic_env/vec_env.hpp"
#include <cmath>

namespace nca::env {

// ============================================================================
// GENERALIST CURRICULUM SCHEDULER
// ============================================================================
// Tracks the progression of the agent. When the agent "solves" the current
// complexity (indicated by a stabilizing advantage / plateauing rewards),
// this scheduler pushes the VecEnv to the next difficulty level.
class CurriculumScheduler {
public:
    CurriculumScheduler(VecEnv* vec_env) 
        : vec_env_(vec_env), current_level_(1), 
          adv_moving_avg_(0.0f), step_count_(0) {}

    // Evaluates the absolute mean advantage of the rollout.
    // If the advantage shrinks to near-zero consistently, the value function 
    // has perfectly mapped the environment, meaning it's time to increase complexity.
    void update_and_step_level(float mean_abs_advantage) {
        const float alpha = 0.05f;
        adv_moving_avg_ = (1.0f - alpha) * adv_moving_avg_ + alpha * mean_abs_advantage;
        step_count_++;

        // Threshold logic for progressive difficulty
        if (step_count_ > 50 && adv_moving_avg_ < 0.05f) {
            current_level_++;
            vec_env_->set_difficulty(current_level_);
            
            // Reset to prevent immediate re-triggering
            step_count_ = 0; 
            adv_moving_avg_ = 1.0f; 
        }
    }

    int get_current_level() const { return current_level_; }

private:
    VecEnv* vec_env_;
    int current_level_;
    float adv_moving_avg_;
    int step_count_;
};

} // namespace nca::env
