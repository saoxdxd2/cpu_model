#include "agentic_env/sim_loop.hpp"
#include "agentic_env/tactical_env.hpp"
#include "core/execution/multimodal_engine.hpp"
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>
#include <torch/torch.h>

using namespace nca::env;
using namespace nca::execution;

int main() {
    try {
        std::cout << "========================================================\n";
        std::cout << " NCA Aether Architecture — Full Agentic Pipeline Target \n";
        std::cout << "========================================================\n\n";

        std::cout << "[1/4] Booting Multimodal Engine (AVX-512)...\n";
        const size_t OBS_DIM = 1616;
        const size_t ACT_DIM = 80;
        nca::config::EngineConfig eng_cfg;
        auto engine = std::make_shared<MultimodalEngine>(OBS_DIM, ACT_DIM, eng_cfg);

        std::cout << "[2/4] Initializing Tactical Environments (32 Parallel)...\n";
        const size_t NUM_ENVS = 32;
        std::vector<std::unique_ptr<Environment>> envs;
        for (size_t i = 0; i < NUM_ENVS; ++i) {
            envs.push_back(std::make_unique<TacticalGridEnv>());
        }
        auto vec_env = std::make_unique<VecEnv>(std::move(envs));

        // Curriculum 1: Dense Survival Rewards
        RewardConfig r_cfg;
        r_cfg.survival_bonus = 1.0f;
        r_cfg.step_penalty = -0.01f;
        r_cfg.win_bonus = 5.0f;
        vec_env->set_reward_config(r_cfg);

        std::cout << "[3/4] Fusing SimLoop & Memory Surface...\n";
        SimLoopConfig sim_cfg;
        sim_cfg.num_envs = NUM_ENVS;
        sim_cfg.rollout_steps = 128;
        sim_cfg.gamma = 0.99f;
        sim_cfg.lambda = 0.95f;
        SimLoop loop(sim_cfg, std::move(vec_env), engine);

        std::cout << "[4/5] Establishing LibTorch PPO Training Bridge...\n";
        
        const int64_t D_MODEL = 1616;
        const int64_t D_ACT = 80;
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
        torch::Tensor policy_weights = torch::randn({D_ACT, D_MODEL}, options);
        auto optimizer = torch::optim::Adam({policy_weights}, torch::optim::AdamOptions(3e-4));

        std::cout << "[5/5] Starting Dual-Phase Adaptation & Training Loop\n\n";
        const int EPOCHS = 10;
        
        for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
            auto t0 = std::chrono::high_resolution_clock::now();
            
            loop.run_epoch();
            
            size_t batch_size = sim_cfg.rollout_steps * sim_cfg.num_envs;
            auto span = loop.get_memory()->get_latest_contiguous(batch_size);
            const float* adv_ptr = loop.get_memory()->get_advantages();
            
            auto t_states = torch::from_blob((void*)span.states, { (int64_t)batch_size, (int64_t)OBS_DIM }, torch::kFloat32).clone();
            auto t_actions = torch::from_blob((void*)span.actions, { (int64_t)batch_size, D_ACT }, torch::kFloat32).clone();
            auto t_adv = torch::from_blob((void*)adv_ptr, { (int64_t)batch_size, 1 }, torch::kFloat32).clone();
            t_adv = torch::clamp(t_adv, -10.0f, 10.0f); // [STABILITY] Advantage Clipping

            optimizer.zero_grad();
            auto logits = t_states.mm(policy_weights.t());       // [B, 80]
            auto log_probs = torch::log_softmax(logits, 1);      // [B, 80]
            auto policy_loss = -(log_probs * t_actions).sum(1, true) * t_adv;
            auto loss = policy_loss.mean();
            
            loss.backward();
            optimizer.step();
            
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            
            std::cout << "Epoch " << std::setw(2) << epoch << " | Loss: " << std::fixed << std::setprecision(4) 
                      << loss.item<float>() << " | Time: " << ms << " ms" << std::endl;
            
            if (epoch == 5) {
                RewardConfig sparse_cfg;
                sparse_cfg.survival_bonus = 0.0f; 
                sparse_cfg.step_penalty = -0.05f; 
                sparse_cfg.win_bonus = 50.0f;     
                loop.get_vec_env()->set_reward_config(sparse_cfg);
            }
        }

        std::cout << "\n[OK] Pipeline Completed Execution with Global Weight Updates.\n";
        return 0;
    } catch(const c10::Error& e) {
        std::cerr << "C10 ERROR: " << e.what() << std::endl;
        return 1;
    } catch(const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    } catch(...) {
        std::cerr << "UNKNOWN EXCEPTION!" << std::endl;
        return 1;
    }
}
