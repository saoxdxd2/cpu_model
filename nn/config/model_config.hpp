#pragma once
// ============================================================================
// NCA — Nano-Core Architecture
// config/model_config.hpp
//
// Single source of truth for ALL model hyperparameters.
// Change a value here, it propagates everywhere.
// ============================================================================

#include <cstdint>

namespace nca::config {

// ═══════════════════════════════════════════════════════════════════════════
// Quantization — OCP Microscaling (MX) Block Format
// ═══════════════════════════════════════════════════════════════════════════
// Block of 32 weights share a single E8M0 (power-of-2) scale byte.
// MXINT8: 32 × int8 + 1 scale = 33 bytes per block (~8.25 bits/weight)
// MXINT4: 32 × int4 + 1 scale = 17 bytes per block (~4.25 bits/weight)
// Activations: dynamic INT8 per-token (VNNI VPDPBUSD requires INT8 operands)
// Recurrent state / inter-core tensors: BF16 or FP32 (NEVER quantized)

constexpr int32_t MX_BLOCK_SIZE = 32;   // Elements per quantization block
constexpr int32_t MXINT8_BITS   = 8;    // Weight precision for Logic/Vision cores
constexpr int32_t MXINT4_BITS   = 4;    // Weight precision for Action core

// ═══════════════════════════════════════════════════════════════════════════
// Shared Dimensions
// ═══════════════════════════════════════════════════════════════════════════

constexpr int32_t D_MODEL      = 2048;  // Hidden dimension (all cores)
constexpr int32_t VOCAB_SIZE    = 32000; // BPE vocabulary
constexpr int32_t MAX_SEQ_LEN   = 4096;  // Maximum sequence length

// ═══════════════════════════════════════════════════════════════════════════
// Vision Core — Adaptive Region Encoder (ARE)
// ═══════════════════════════════════════════════════════════════════════════

constexpr int32_t IMAGE_SIZE          = 384;   // Input image: 384×384
constexpr int32_t VISION_STEM_CHANNELS = 64;   // Initial conv channels
constexpr int32_t VISION_HIDDEN_DIM    = 512;  // Internal vision dim (projected to D_MODEL at output)
constexpr int32_t VISION_FEATURE_GRID  = 24;   // After downsampling: 24×24 = 576 positions

// E-AdaPrune spectral token budget
constexpr int32_t SPECTRAL_MIN_TOKENS = 1;     // Blank wall → 1 token
constexpr int32_t SPECTRAL_MAX_TOKENS = 64;    // Dense UI → 64 tokens
constexpr float   SPECTRAL_THRESHOLD  = 0.95f; // τ: capture 95% of spectral energy

// ═══════════════════════════════════════════════════════════════════════════
// Logic Core — Triple-Hybrid Recurrence
// ═══════════════════════════════════════════════════════════════════════════

constexpr int32_t LOGIC_LAYERS     = 24;    // Total layers
constexpr int32_t LOGIC_MLP_RATIO  = 267;   // ×2.67 → MLP dim = D_MODEL * 267 / 100
constexpr int32_t LOGIC_MLP_DIM    = (D_MODEL * LOGIC_MLP_RATIO) / 100; // = 5461

// Gated Linear Recurrence (GLR) — Block Type A
constexpr int32_t GLR_STATE_DIM    = D_MODEL; // Diagonal gate, same as hidden dim

// Selective SSM — Block Type B
constexpr int32_t SSM_STATE_DIM    = 64;    // State dim per head
constexpr int32_t SSM_NUM_HEADS    = 32;    // Number of SSM heads

// Sparse Local Attention — Block Type C
constexpr int32_t SLA_WINDOW_SIZE  = 256;   // Sliding window width
constexpr int32_t SLA_NUM_HEADS    = 16;    // Attention heads for SLA layers
constexpr int32_t SLA_HEAD_DIM     = D_MODEL / SLA_NUM_HEADS; // = 128

// Layer schedule: which layers get which block type
// Every 4th layer: SSM. Every 8th layer: SSM + SLA. Rest: GLR.
// Layers with SSM:      3, 7, 11, 15, 19, 23
// Layers with SSM+SLA:  7, 15, 23
constexpr int32_t SSM_INTERVAL = 4;        // SSM every N layers
constexpr int32_t SLA_INTERVAL = 8;        // SLA every M layers

// Adaptive Computation Time (ACT) — L/H Cycling
constexpr int32_t MAX_ACT_CYCLES    = 4;    // Max re-entry cycles (24→96 effective layers)
constexpr float   ACT_HALT_THRESHOLD = 0.5f; // Halt when confidence > θ
constexpr float   ACT_PONDER_COST    = 0.01f; // λ: regularization weight for ponder cost

// ═══════════════════════════════════════════════════════════════════════════
// Action Core — Fast Output Generator
// ═══════════════════════════════════════════════════════════════════════════

constexpr int32_t ACTION_LAYERS    = 16;    // Fewer layers (no need to "think")
constexpr int32_t ACTION_MLP_DIM   = LOGIC_MLP_DIM; // Same MLP ratio

// ═══════════════════════════════════════════════════════════════════════════
// Latent Communication
// ═══════════════════════════════════════════════════════════════════════════

constexpr int32_t ADAPTER_DIM      = D_MODEL; // Adapter input = output = D_MODEL
// Adapter is a single linear: [D_MODEL, D_MODEL] = 2048×2048 = ~16MB FP32

// ═══════════════════════════════════════════════════════════════════════════
// Special Token IDs (placeholder — finalized with tokenizer)
// ═══════════════════════════════════════════════════════════════════════════

constexpr int32_t TOKEN_PAD   = 0;
constexpr int32_t TOKEN_BOS   = 1;
constexpr int32_t TOKEN_EOS   = 2;
constexpr int32_t TOKEN_THINK = 3;  // <think> — latent reasoning (not emitted)
constexpr int32_t TOKEN_ANSWER = 4; // <answer> — start visible output

} // namespace nca::config
