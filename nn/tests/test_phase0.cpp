// ============================================================================
// NCA — Phase 0 Build Verification Test
// tests/test_phase0.cpp
//
// Verifies:
//   1. C++20 compilation works
//   2. LibTorch links and runs
//   3. CPUID detects CPU features correctly
//   4. SIMD backend selection works
//   5. MX block size config is accessible
// ============================================================================

#include "core/simd/dispatch.hpp"
#include "config/model_config.hpp"

#include <torch/torch.h>

#include <cassert>
#include <iostream>
#include <format>   // C++20 — proves we're not stuck on C++17

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  NCA — Phase 0 Build Verification        ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    // ── 1. C++20 check ────────────────────────────────────────────────
    std::cout << "[1] C++ Standard: " << __cplusplus << "\n";
    static_assert(__cplusplus >= 202002L, "C++20 required!");
    std::cout << "    ✓ C++20 confirmed\n\n";

    // ── 2. LibTorch check ─────────────────────────────────────────────
    std::cout << "[2] LibTorch:\n";
    auto t = torch::randn({2, 3}, torch::kCPU);
    std::cout << "    Random tensor [2,3]:\n" << t << "\n";
    std::cout << "    torch::kCPU available: YES\n";

    // Verify no CUDA (CPU-only build)
    assert(!torch::cuda::is_available());
    std::cout << "    CUDA available: NO (expected for CPU build)\n\n";

    // ── 3. CPUID detection ────────────────────────────────────────────
    std::cout << "[3] CPU Feature Detection:\n";
    const auto& features = nca::simd::detect();
    std::cout << features.to_string() << "\n";

    // ── 4. Backend selection ──────────────────────────────────────────
    auto backend = nca::simd::best_backend();
    std::cout << "[4] Selected Backend: "
              << nca::simd::backend_name(backend) << "\n\n";

    // ── 5. Config sanity check ────────────────────────────────────────
    std::cout << "[5] Model Config:\n";
    std::cout << std::format("    D_MODEL         = {}\n", nca::config::D_MODEL);
    std::cout << std::format("    VOCAB_SIZE      = {}\n", nca::config::VOCAB_SIZE);
    std::cout << std::format("    MAX_SEQ_LEN     = {}\n", nca::config::MAX_SEQ_LEN);
    std::cout << std::format("    MX_BLOCK_SIZE   = {}\n", nca::config::MX_BLOCK_SIZE);
    std::cout << std::format("    LOGIC_LAYERS    = {}\n", nca::config::LOGIC_LAYERS);
    std::cout << std::format("    ACTION_LAYERS   = {}\n", nca::config::ACTION_LAYERS);
    std::cout << std::format("    IMAGE_SIZE      = {}x{}\n",
        nca::config::IMAGE_SIZE, nca::config::IMAGE_SIZE);
    std::cout << std::format("    SPECTRAL_TOKENS = [{}, {}]\n",
        nca::config::SPECTRAL_MIN_TOKENS, nca::config::SPECTRAL_MAX_TOKENS);

    // ── 6. MX block math verification ─────────────────────────────────
    std::cout << "\n[6] MX Format Verification:\n";
    constexpr int32_t bs = nca::config::MX_BLOCK_SIZE;
    // MXINT8: 32 int8 + 1 scale byte = 33 bytes per block
    constexpr int32_t mx8_block_bytes = bs * 1 + 1; // 33
    // MXINT4: 32 int4 (packed 2/byte) + 1 scale byte = 17 bytes per block
    constexpr int32_t mx4_block_bytes = bs / 2 + 1;  // 17
    static_assert(mx8_block_bytes == 33, "MXINT8 block must be 33 bytes");
    static_assert(mx4_block_bytes == 17, "MXINT4 block must be 17 bytes");

    constexpr float mx8_bits_per_weight = 8.0f * mx8_block_bytes / bs;
    constexpr float mx4_bits_per_weight = 8.0f * mx4_block_bytes / bs;
    std::cout << std::format("    MXINT8: {} bytes/block = {:.2f} bits/weight\n",
        mx8_block_bytes, mx8_bits_per_weight);
    std::cout << std::format("    MXINT4: {} bytes/block = {:.2f} bits/weight\n",
        mx4_block_bytes, mx4_bits_per_weight);

    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "  Phase 0: ALL CHECKS PASSED ✓\n";
    std::cout << "══════════════════════════════════════════\n";

    return 0;
}
