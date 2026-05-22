#pragma once
#include "agentic_env/env.hpp"
#include <string>
#include <vector>
#include <filesystem>

namespace nca::env {

/**
 * VSCodeEnv
 * A high-fidelity agentic environment wrapping a VSCode workspace.
 * Modalities: 
 *  - Alphabet (Raw code streams)
 *  - Vision (GUI screenshots/UI state)
 *  - Terminal (Standard out/err)
 */
class VSCodeEnv : public Environment {
public:
    explicit VSCodeEnv(const std::string& workspace_root);

    // Standard RL Interface
    void reset(float* obs_out) override;
    StepResult step(const float* action_in, float* obs_out) override;
    void get_action_mask(float* mask_out) const override;

    // Dimensions
    size_t get_observation_dim() const override { return 1616; }
    size_t get_action_dim() const override { return 80; }

    // Custom Visual/Alphabet Ingestion
    void render_vision(float* vision_out);
    void stream_code(float* alphabet_out);
    void stream_terminal(float* terminal_out);

    // Agentic Actions
    void apply_code_patch(const std::string& filename, const std::string& patch);

private:
    std::string root_;
    std::vector<std::string> active_files_;
    size_t cursor_pos_;
    
    // Internal simulator state
    float health_;
    float compile_status_;
    std::string terminal_output_; // Captures real stdout/stderr

    void execute_filesystem_op(int action_id, float intensity);
    void execute_shell_command(const std::string& cmd);
    void update_simulation_state();
};

} // namespace nca::env
