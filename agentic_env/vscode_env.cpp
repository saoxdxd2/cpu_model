#include "agentic_env/vscode_env.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>

namespace nca::env {

VSCodeEnv::VSCodeEnv(const std::string& workspace_root) 
    : root_(workspace_root), cursor_pos_(0), health_(1.0f), compile_status_(1.0f) {
}

void VSCodeEnv::reset(float* obs_out) {
    active_files_.clear();
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_)) {
        if (entry.is_regular_file() && active_files_.size() < 10) {
            active_files_.push_back(entry.path().string());
        }
    }
    cursor_pos_ = 0;
    health_ = 1.0f;
    std::memset(obs_out, 0, get_observation_dim() * sizeof(float));
}

StepResult VSCodeEnv::step(const float* action_in, float* obs_out) {
    // 1. Parse Agentic Intent
    int command_id = 0;
    float max_v = action_in[0];
    for(int i=1; i<5; ++i) {
        if(action_in[i] > max_v) { max_v = action_in[i]; command_id = i; }
    }

    // 2. Execute on Silicon Workspace
    execute_filesystem_op(command_id, max_v);
    update_simulation_state();

    // 3. Multimodal Observation Fusion
    std::memset(obs_out, 0, 1616 * sizeof(float));
    stream_code(obs_out);             // First 800 floats: Source Code
    stream_terminal(obs_out + 800);   // Next 800 floats: Terminal Output

    StepResult res;
    res.reward = (compile_status_ > 0.5f) ? 0.1f : -1.0f;
    res.terminated = (health_ <= 0.0f);
    res.truncated = false;
    
    return res;
}

void VSCodeEnv::get_action_mask(float* mask_out) const {
    std::memset(mask_out, 0, get_action_dim() * sizeof(float));
}

void VSCodeEnv::execute_filesystem_op(int action_id, float intensity) {
    const char* COMMAND_REGISTRY[] = {
        "echo IDLE",
        "git status",
        "npm list --depth=0",
        "tsc --version",
        "git diff"
    };

    if (intensity > 0.7f && action_id < 5) {
        std::cout << "  [ENV] Executing Live Command: " << COMMAND_REGISTRY[action_id] << "\n";
        execute_shell_command(COMMAND_REGISTRY[action_id]);
    }

    switch(action_id) {
        case 1: // READ
            cursor_pos_ = (cursor_pos_ + 128); 
            break;
        case 2: // WRITE
            compile_status_ *= 0.99f; 
            break;
        case 3: // TEST
            if (intensity > 0.8f) compile_status_ = 1.0f;
            break;
    }
}

void VSCodeEnv::execute_shell_command(const std::string& cmd) {
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) return;

    char buffer[128];
    terminal_output_ = "";
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        terminal_output_ += buffer;
    }
    
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif

    // Clip output to save memory
    if (terminal_output_.length() > 1024) terminal_output_ = terminal_output_.substr(0, 1024);
}

void VSCodeEnv::update_simulation_state() {
    if (compile_status_ < 0.2f) health_ -= 0.01f;
}

void VSCodeEnv::render_vision(float* vision_out) {
    // Generates a synthetic UI map for the SiliconVisionEncoder
    std::memset(vision_out, 0, 256 * 256 * sizeof(float));
    for(int i=0; i<100; ++i) vision_out[i * 256 + 10] = 1.0f; 
    if (compile_status_ < 0.5f) vision_out[50 * 256 + 50] = 1.0f; 
}

void VSCodeEnv::stream_code(float* alphabet_out) {
    // Streams the raw alphabet primitives from the current cursor
    // For now, simulated as constant signal
    for(int i=0; i<800; ++i) alphabet_out[i] = 0.5f;
}

void VSCodeEnv::apply_code_patch(const std::string& filename, const std::string& patch) {
    std::string full_path = root_ + "/" + filename;
    std::cout << "  [ENV] Applying Silicon Patch to: " << filename << "\n";
    
    std::ofstream out(full_path, std::ios::app); // Surgical append for now
    if (out.is_open()) {
        out << "\n" << patch << "\n";
        out.close();
        compile_status_ = 1.0f; // Assuming the patch fixed the issue
    }
}

void VSCodeEnv::stream_terminal(float* terminal_out) {
    // Map terminal characters to alphabet-level primitives
    for (size_t i = 0; i < std::min(terminal_output_.length(), (size_t)800); ++i) {
        terminal_out[i] = static_cast<float>(terminal_output_[i]) / 255.0f;
    }
}

} // namespace nca::env
