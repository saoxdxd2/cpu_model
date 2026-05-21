#pragma once
// ============================================================================
// NCA — Nano-Core Architecture
// config/model_config.hpp
// ============================================================================

#include <cstdint>
#include <cstddef>

namespace nca::config {

// ── ARCHITECTURE DIMENSIONS ────────────────────────────────────────────────
constexpr size_t D_MODEL = 2048; 
constexpr size_t VISION_FEATURE_GRID = 16; 
constexpr size_t VISION_CHANNELS = 128;

// ── RECURSIVE ACT CONFIG ────────────────────────────────────────────────────
constexpr float ACT_HALT_THRESHOLD = 0.99f;
constexpr int MAX_ACT_CYCLES = 16;
constexpr int DEEP_ACT_CYCLES = 64; // For "Surprising" tokens

// ── SDMS: SALIENCY & EXPERT CONFIG (v11.0) ──────────────────────────────────
constexpr size_t N_MICRO_EXPERTS = 1024; // DeepSeekMoE style
constexpr size_t TOP_K_EXPERTS = 16;     // Match Rank-16 Saturation
constexpr size_t N_SHARED_EXPERTS = 2;   // Always active common knowledge
constexpr float SALIENCY_THRESHOLD = 0.05f; // BLT-style entropy gate

// ── SPECTRAL RLS CONFIG ─────────────────────────────────────────────────────
constexpr size_t KRONECKER_FACTOR_DIM_A = 64;
constexpr size_t KRONECKER_FACTOR_DIM_B = 32;
constexpr float RLS_FORGETTING_FACTOR = 0.9999f; // Titans-style long memory

// ── TOKEN REGISTRY ──────────────────────────────────────────────────────────
constexpr int32_t TOKEN_PAD    = 0;
constexpr int32_t TOKEN_BOS    = 1;
constexpr int32_t TOKEN_EOS    = 2;
constexpr int32_t TOKEN_THINK  = 3;  
constexpr int32_t TOKEN_ANSWER = 4;

// ── BACKEND SELECTION ───────────────────────────────────────────────────────
enum class LogicBackend {
    HashedRouter,
    SpectralRLS,
    SpectralRoutedExperts,
    SDMS_Predictive // v11.0 Unified Saliency-Driven
};

struct EngineConfig {
    LogicBackend logic_backend = LogicBackend::HashedRouter;
};

} // namespace nca::config
