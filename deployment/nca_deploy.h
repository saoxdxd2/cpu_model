#ifndef NCA_DEPLOY_H
#define NCA_DEPLOY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * NCA Deployment C-API
 * Stable, high-performance interface for production environments.
 */

#ifdef _WIN32
  #ifdef NCA_DEPLOY_EXPORT
    #define NCA_API __declspec(dllexport)
  #else
    #define NCA_API __declspec(dllimport)
  #endif
#else
  #define NCA_API __attribute__((visibility("default")))
#endif

typedef struct nca_engine_t* nca_engine_ptr;

// Lifecycle
NCA_API nca_engine_ptr nca_engine_create(size_t obs_dim, size_t act_dim);
NCA_API void           nca_engine_destroy(nca_engine_ptr engine);

// State Management
NCA_API void nca_engine_load_pt(nca_engine_ptr engine, const char* path);
NCA_API void nca_engine_reset(nca_engine_ptr engine);

// Introspection: Returns a JSON string describing the "Silicon Wiring"
// Memory is owned by the engine, do not free.
NCA_API const char* nca_get_topology(nca_engine_ptr engine);

// Inference
// Executes one high-speed step. input/output must be pre-allocated.
NCA_API void nca_engine_step(nca_engine_ptr engine, 
                             const float* observation, 
                             float* actions_out);

// Exposes the 2048-dimensional continuous graph state (memory curve)
NCA_API void nca_engine_get_state(nca_engine_ptr engine, float* state_out);

#ifdef __cplusplus
}
#endif

#endif // NCA_DEPLOY_H
