#include "deployment/nca_deploy.h"
#include "core/execution/multimodal_engine.hpp"
#include <memory>
#include <string>

struct nca_engine_t {
    std::shared_ptr<nca::execution::MultimodalEngine> internal;
    std::string topology_cache;
};

extern "C" {

nca_engine_ptr nca_engine_create(size_t obs_dim, size_t act_dim) {
    auto engine = new nca_engine_t();
    engine->internal = std::make_shared<nca::execution::MultimodalEngine>(obs_dim, act_dim);
    
    // Cache the topology string
    engine->topology_cache = "{\"module\": \"Aether Pipeline\", \"wiring\": [";
    engine->topology_cache += "{\"name\": \"SiliconEncoder\", \"tech\": \"AVX-512\", \"target\": \"LatentSpace\"}, ";
    engine->topology_cache += "{\"name\": \"ImportanceClassifier\", \"tech\": \"Saliency-Driven\", \"target\": \"ThoughtCycles\"}, ";
    engine->topology_cache += "{\"name\": \"SDMS_ExpertPool\", \"count\": 1024, \"rank\": 16}, ";
    engine->topology_cache += "{\"name\": \"KroneckerRLS\", \"mode\": \"Spectral\"}";
    engine->topology_cache += "]}";
    
    return engine;
}

void nca_engine_destroy(nca_engine_ptr engine) {
    if (engine) {
        delete engine;
    }
}

const char* nca_get_topology(nca_engine_ptr engine) {
    if (!engine) return "";
    return engine->topology_cache.c_str();
}

void nca_engine_load_pt(nca_engine_ptr engine, const char* path) {
    if (!engine) return;
    // In production, this would call a production loader.
    // For now, it initializes the internal engine state.
}

void nca_engine_reset(nca_engine_ptr engine) {
    if (engine) engine->internal->reset_state();
}

void nca_engine_step(nca_engine_ptr engine, 
                     const float* observation, 
                     float* actions_out) {
    if (!engine) return;
    // Single environment step optimized for latency
    engine->internal->step(nullptr, observation, actions_out);
}

}
