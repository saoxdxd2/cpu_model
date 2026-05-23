#pragma once
#include "core/execution/multimodal_engine.hpp"
#include "agentic_env/replay_memory.hpp"
#include <torch/torch.h>
#include <memory>

namespace nca::training {

struct PPOConfig {
    float lr = 1e-4;
    float gamma = 0.99;
    float lambda = 0.95;
    float clip_epsilon = 0.2;
    float c_value = 0.5;
    float c_entropy = 0.01;
    int k_epochs = 4;
    size_t batch_size = 64;
};

/**
 * PPOTrainer
 * High-performance PPO implementation for the NCA architecture.
 * Implements the Proximal Policy Optimization clipping objective.
 */
class PPOTrainer {
public:
    PPOTrainer(std::shared_ptr<nca::execution::MultimodalEngine> engine, PPOConfig cfg = {});

    // Performs multiple PPO update epochs on the provided trajectory data
    void train_on_memory(nca::env::ReplayMemory& memory);

private:
    std::shared_ptr<nca::execution::MultimodalEngine> engine_;
    PPOConfig cfg_;

    // Tensors for policy and value heads (Shared with MultimodalEngine via WeightRegistry)
    torch::Tensor policy_weights_;
    torch::Tensor value_weights_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;

    void update_step(torch::Tensor states, torch::Tensor actions, 
                     torch::Tensor log_probs_old, torch::Tensor returns, 
                     torch::Tensor advantages);
};

} // namespace nca::training
