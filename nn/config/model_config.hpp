#pragma once
// ============================================================================
// NCA — Nano-Core Architecture
// config/model_config.hpp
// ============================================================================

#include <cstdint>
#include <cstddef>

namespace nca::config {

// ── ARCHITECTURE DIMENSIONS ────────────────────────────────────────────────
constexpr size_t D_MODEL = 2048; // Must be power of 2 for FWHT
constexpr size_t VISION_FEATURE_GRID = 16; 
constexpr size_t VISION_CHANNELS = 128;

// ── RECURSIVE ACT CONFIG ────────────────────────────────────────────────────
constexpr float ACT_HALT_THRESHOLD = 0.99f;
constexpr int MAX_ACT_CYCLES = 16;

// ── SPECTRAL RLS CONFIG (v7.0) ──────────────────────────────────────────────
constexpr size_t SPECTRAL_BLOCK_SIZE = 2048; // Must match D_MODEL
constexpr size_t KRONECKER_FACTOR_DIM = 45;   // ~sqrt(2048)
constexpr float RLS_DAMPING = 1e-4f;
constexpr float RLS_FORGETTING_FACTOR = 0.999f;

// ── TOKEN REGISTRY ──────────────────────────────────────────────────────────
constexpr int32_t TOKEN_PAD    = 0;
constexpr int32_t TOKEN_BOS    = 1;
constexpr int32_t TOKEN_EOS    = 2;
constexpr int32_t TOKEN_THINK  = 3;  
constexpr int32_t TOKEN_ANSWER = 4;

// ── BACKEND SELECTION ───────────────────────────────────────────────────────
enum class LogicBackend {
    HashedRouter, // v6.2 Default
    SpectralRLS   // v7.0 Opt-in
};

struct EngineConfig {
    LogicBackend logic_backend = LogicBackend::HashedRouter;
};

} // namespace nca::config
