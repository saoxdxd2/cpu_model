#include "agentic_env/env.hpp"
#include "agentic_env/vec_env.hpp"
#include "agentic_env/replay_memory.hpp"
#include <iostream>
#include <chrono>
#include <cassert>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>

using namespace nca::env;

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << std::endl; g_fail++; } \
} while(0)

class DummyEnv : public Environment {
public:
    DummyEnv(size_t obs, size_t act) : o_(obs), a_(act), t_(0) {}
    void reset(float* o) override { t_=0; for(size_t i=0;i<o_;++i) o[i]=0.f; }
    StepResult step(const float* a, float* o) override {
        t_++; for(size_t i=0;i<o_;++i) o[i]=float(t_)+a[i%a_];
        return {1.f, t_>=100, false};
    }
    size_t get_observation_dim() const override { return o_; }
    size_t get_action_dim() const override { return a_; }
private:
    size_t o_, a_; int t_;
};

void test_data_integrity() {
    ReplayMemory rm(8, 4, 2);
    float s[4]={1,2,3,4}, a[2]={10,20};
    rm.push(s, a, 42.f, false);
    BatchSpan sp[2];
    size_t n = rm.get_latest_spans(1, sp);
    CHECK(n==1 && sp[0].count==1, "Span count");
    CHECK(sp[0].states[0]==1.f && sp[0].states[3]==4.f, "State data");
    CHECK(sp[0].actions[0]==10.f, "Action data");
    CHECK(sp[0].rewards[0]==42.f, "Reward");
    CHECK(sp[0].dones[0]==false, "Done");
    std::cout << "[OK] Data integrity." << std::endl;
}

void test_interleaved_gae() {
    const size_t ENVS = 2;
    const size_t STEPS = 3;
    ReplayMemory rm(16, 4, 1);
    
    // Env 0: r = [1, 2, 3], v = [10, 9, 8], done = [0, 0, 1]
    // Env 1: r = [4, 5, 6], v = [20, 19, 18], done = [0, 0, 0]
    
    for(size_t t=0; t<STEPS; ++t) {
        float s0[4]={0}, a0[1]={0};
        rm.push(s0, a0, t+1.0f, t==2); // env 0
        
        float s1[4]={0}, a1[1]={0};
        rm.push(s1, a1, t+4.0f, false); // env 1
    }
    
    float vals[6] = {10, 20, 9, 19, 8, 18};
    rm.write_values(6, vals);
    
    float next_v[2] = {0, 17}; // Env 0 is terminal (next_v=0), Env 1 continues (next_v=17)
    rm.fused_compute_gae(STEPS, ENVS, 0.99f, 0.95f, next_v);
    
    const float* adv = rm.get_advantages();
    
    // Env 1 backward sweep:
    // t=2: delta = 6 + 0.99*17 - 18 = 6 + 16.83 - 18 = 4.83. A = 4.83
    // t=1: delta = 5 + 0.99*18 - 19 = 5 + 17.82 - 19 = 3.82. A = 3.82 + 0.99*0.95*4.83 = 3.82 + 4.542615 = 8.362615
    // t=0: delta = 4 + 0.99*19 - 20 = 4 + 18.81 - 20 = 2.81. A = 2.81 + 0.99*0.95*8.362615 = 2.81 + 7.865... = 10.675
    
    // Values are at interleaved indices:
    // Env 1 is at indices 1, 3, 5.
    CHECK(std::abs(adv[5] - 4.83f) < 0.01f, "Env1 t=2 GAE");
    CHECK(std::abs(adv[3] - 8.36f) < 0.01f, "Env1 t=1 GAE");
    CHECK(std::abs(adv[1] - 10.67f) < 0.01f, "Env1 t=0 GAE");
    
    std::cout << "[OK] Interleaved GAE verified (A[1]=" << adv[1] << ")." << std::endl;
}

void perf_push_batch() {
    // 50% less bandwidth
    const size_t CAP=1'000'000, OBS=512, ACT=16, B=32;
    ReplayMemory rm(CAP, OBS, ACT);
    float* st = static_cast<float*>(alloc_aligned(B*OBS*sizeof(float),64));
    float* ac = static_cast<float*>(alloc_aligned(B*ACT*sizeof(float),64));
    float* rw = static_cast<float*>(alloc_aligned(B*sizeof(float),64));
    bool*  dn = static_cast<bool*>(alloc_aligned(B*sizeof(bool),64));
    std::memset(st,0,B*OBS*sizeof(float)); std::memset(ac,0,B*ACT*sizeof(float));
    std::memset(rw,0,B*sizeof(float)); std::memset(dn,0,B*sizeof(bool));
    
    auto t0 = std::chrono::high_resolution_clock::now();
    const int ITERS = 100'000;
    for(int i=0;i<ITERS;++i) rm.push_batch(B, st, ac, rw, dn);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
    
    // New exact byte calculation without next_states
    size_t bpi = B*(OBS+ACT+1)*sizeof(float)+B*sizeof(bool);
    double gb = (double(bpi)*ITERS/(1024.*1024.*1024.))/(ms/1000.);
    std::cout << "[PERF] push_batch (No Next-States): " << gb << " GB/s (" << ms << " ms)" << std::endl;
    free_aligned(st); free_aligned(ac); free_aligned(rw); free_aligned(dn);
}

void perf_interleaved_gae() {
    const size_t CAP=65536, OBS=4, ACT=1;
    const size_t ENVS = 32;
    const size_t STEPS = CAP / ENVS; // 2048 steps
    ReplayMemory rm(CAP, OBS, ACT);
    
    float s[4]={}, a[1]={};
    for(size_t i=0;i<CAP;++i) rm.push(s, a, 1.f, false);
    
    float* vals = static_cast<float*>(alloc_aligned(CAP*sizeof(float), 64));
    rm.write_values(CAP, vals);
    float next_v[ENVS] = {0};
    
    auto t0 = std::chrono::high_resolution_clock::now();
    const int ITERS = 100'000;
    for(int i=0; i<ITERS; ++i) {
        rm.fused_compute_gae(STEPS, ENVS, 0.99f, 0.95f, next_v);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
    double ns_per_step = (ms * 1e6) / (double(ITERS) * CAP);
    
    std::cout << "[PERF] Interleaved GAE: " << ns_per_step << " ns/step (" << ms << " ms)" << std::endl;
    free_aligned(vals);
}

int main() {
    std::cout << "=== Agentic Layer Validation ===" << std::endl;
    test_data_integrity();
    test_interleaved_gae();
    
    if (g_fail > 0) return 1;
    
    std::cout << "\n=== Performance ===" << std::endl;
    perf_push_batch();
    perf_interleaved_gae();
    return 0;
}
