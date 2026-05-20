// ============================================================================
// NCA — Phase 3: MXINT8 VNNI Math Core Test
// tests/test_phase3_vnni.cpp
//
// Proof-of-Concept testing for AVX-512 VNNI using OCP MX E8M0 Block scaling.
// ============================================================================

#include "core/simd/dispatch.hpp"
#include <torch/torch.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <bit>

namespace nca::quant {

// OCP MX Structure Arrays for contiguous memory loads
struct MXINT8Tensor {
    std::vector<int8_t> data;
    std::vector<uint8_t> scales;
};

struct MXUINT8Tensor {
    std::vector<uint8_t> data;
    std::vector<uint8_t> scales;
};

// Extract E8M0 scale directly into IEEE-754 biased exponent
inline uint8_t extract_e8m0(float max_abs) {
    if (max_abs == 0.0f) return 0;
    float scale = max_abs / 127.0f;
    uint32_t bits = std::bit_cast<uint32_t>(scale);
    uint32_t mantissa = bits & 0x7FFFFF;
    uint8_t exp = (bits >> 23) & 0xFF;
    if (mantissa > 0) exp += 1; // Round up scale
    return exp;
}

// Rehydrate E8M0 scale back into IEEE-754 float
inline float decode_scale(uint8_t e) {
    if (e == 0) return 0.0f;
    uint32_t bits = static_cast<uint32_t>(e) << 23;
    return std::bit_cast<float>(bits);
}

void quantize_w(const float* in, MXINT8Tensor& out, size_t num_blocks) {
    out.data.resize(num_blocks * 32);
    out.scales.resize(num_blocks);
    
    for (size_t b = 0; b < num_blocks; ++b) {
        float max_abs = 0;
        for (int i=0; i<32; ++i) max_abs = std::max(max_abs, std::abs(in[b*32 + i]));
        
        out.scales[b] = extract_e8m0(max_abs);
        float scale = decode_scale(out.scales[b]);
        float inv = scale > 0 ? 1.0f / scale : 0.0f;
        
        for (int i=0; i<32; ++i) {
            float v = std::round(in[b*32 + i] * inv);
            out.data[b*32 + i] = static_cast<int8_t>(std::clamp(v, -127.0f, 127.0f));
        }
    }
}

void quantize_x(const float* in, MXUINT8Tensor& out, size_t num_blocks) {
    out.data.resize(num_blocks * 32);
    out.scales.resize(num_blocks);
    
    for (size_t b = 0; b < num_blocks; ++b) {
        float max_abs = 0;
        // For Phase 3 basic proof, we assume positive X activations (e.g. ReLU)
        for (int i=0; i<32; ++i) max_abs = std::max(max_abs, std::abs(in[b*32 + i])); 
        
        out.scales[b] = extract_e8m0(max_abs);
        float scale = decode_scale(out.scales[b]);
        float inv = scale > 0 ? 1.0f / scale : 0.0f;
        
        for (int i=0; i<32; ++i) {
            float v = std::round(std::abs(in[b*32 + i]) * inv);
            out.data[b*32 + i] = static_cast<uint8_t>(std::clamp(v, 0.0f, 127.0f)); 
        }
    }
}

// ----------------------------------------------------------------------------
// Phase 3 Proof-of-Concept: VNNI Dot Product using MX Quantization
// ----------------------------------------------------------------------------
#if defined(__AVX512VNNI__) || defined(_MSC_VER)
float vnni_dot_product(const MXINT8Tensor& W, const MXUINT8Tensor& X, size_t num_blocks) {
    float global_sum = 0.0f;
    size_t b = 0;
    
    // Process 64 elements (2 blocks) per cycle
    for (; b <= num_blocks - 2; b += 2) [[likely]] {
        __m512i vW = _mm512_loadu_si512((const __m512i*)&W.data[b * 32]);
        __m512i vX = _mm512_loadu_si512((const __m512i*)&X.data[b * 32]);
        
        // VNNI Magic: 64 MACs into 16 32-bit accumulators
        __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), vX, vW);
        __m512 f_acc = _mm512_cvtepi32_ps(acc);
        
        // Scale Rehydration
        float s0 = decode_scale(W.scales[b]) * decode_scale(X.scales[b]);
        float s1 = decode_scale(W.scales[b+1]) * decode_scale(X.scales[b+1]);
        
        __m256 v_s0 = _mm256_set1_ps(s0);
        __m256 v_s1 = _mm256_set1_ps(s1);
        __m512 v_scales = _mm512_insertf32x8(_mm512_castps256_ps512(v_s0), v_s1, 1);
        
        f_acc = _mm512_mul_ps(f_acc, v_scales);
        global_sum += _mm512_reduce_add_ps(f_acc);
    }
    
    if (b < num_blocks) [[unlikely]] {
        __m256i vW = _mm256_loadu_si256((const __m256i*)&W.data[b * 32]);
        __m256i vX = _mm256_loadu_si256((const __m256i*)&X.data[b * 32]);
        __m256i acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vX, vW);
        __m256 f_acc = _mm256_cvtepi32_ps(acc);
        float s = decode_scale(W.scales[b]) * decode_scale(X.scales[b]);
        f_acc = _mm256_mul_ps(f_acc, _mm256_set1_ps(s));
        
        __m128 v_low = _mm_add_ps(_mm256_castps256_ps128(f_acc), _mm256_extractf128_ps(f_acc, 1));
        v_low = _mm_hadd_ps(v_low, v_low);
        v_low = _mm_hadd_ps(v_low, v_low);
        global_sum += _mm_cvtss_f32(v_low);
    }
    
    return global_sum;
}
#endif

} // namespace nca::quant

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  NCA — Phase 3: MX VNNI Dot Product      ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    constexpr size_t D = 8192; // Typical width
    constexpr size_t BLOCKS = D / 32;
    
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor W_ref = torch::randn({D}, opts);
    torch::Tensor X_ref = torch::relu(torch::randn({D}, opts)); // Relu for uint8 mapping test

    float py_dot = torch::dot(W_ref, X_ref).item<float>();

    // Dynamic MX Quantization
    nca::quant::MXINT8Tensor W_q;
    nca::quant::MXUINT8Tensor X_q;
    
    nca::quant::quantize_w(W_ref.data_ptr<float>(), W_q, BLOCKS);
    nca::quant::quantize_x(X_ref.data_ptr<float>(), X_q, BLOCKS);

    std::cout << "[1] Correctness Verification\n";
    
    float vnni_dot = nca::quant::vnni_dot_product(W_q, X_q, BLOCKS);
    float err = std::abs(py_dot - vnni_dot) / std::abs(py_dot); // Relative error

    std::cout << "    LibTorch FP32 : " << py_dot << "\n";
    std::cout << "    VNNI MXINT8   : " << vnni_dot << "\n";
    std::cout << "    Relative Error: " << (err * 100.0f) << "%\n";

    if (err > 0.05f) { // 5% quantization error threshold
        std::cerr << "\n❌ Quantization loss is too high (>5%). VNNI scaling logic failed.\n";
        return 1;
    }
    std::cout << "    ✓ VNNI successfully preserved mathematical structure.\n\n";

    // 2. Benchmarking
    std::cout << "[2] Micro-Benchmark (10,000 iterations of D=8192)\n";
    constexpr int ITERS = 10000;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        volatile float d = torch::dot(W_ref, X_ref).item<float>();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_py = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        volatile float d = nca::quant::vnni_dot_product(W_q, X_q, BLOCKS);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double ms_vnni = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "    LibTorch FP32 : " << ms_py << " ms\n";
    std::cout << "    MX VNNI Dot   : " << ms_vnni << " ms (" << (ms_py/ms_vnni) << "x speedup)\n";

    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "  Phase 3: VNNI Test PASS ✓\n";
    std::cout << "══════════════════════════════════════════\n";

    return 0;
}
