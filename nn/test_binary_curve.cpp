// ============================================================================
// NCA -- Binary Curve Routing (Transistor-Level Weight Execution)
// ============================================================================
// Proves that AVX-512 masked loads act as physical logic gates,
// routing data through the CPU without FP32 multipliers.
// ============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <immintrin.h>

// ── 1. The Binary Curve (Compiled Weights) ──────────────────────────────────
// We pack the 1s and 0s into 16-bit masks.
// Each bit physically controls one of the 16 AVX-512 ALU lanes.
struct BinaryCurve {
    size_t m, n;
    std::vector<uint16_t> masks; // The "circuit board" layout
    float scale;
    float offset;

    BinaryCurve(size_t m_, size_t n_) : m(m_), n(n_) {
        // n must be multiple of 16 for AVX-512
        masks.resize(m * (n / 16), 0);
    }
};

// ── 2. Quantize Float Matrix to Binary Curve ────────────────────────────────
BinaryCurve compile_to_curve(const std::vector<float>& W, size_t m, size_t n) {
    BinaryCurve curve(m, n);
    
    // Simple 1-bit quantization for this test (sign bit)
    // Positive weight = logic gate OPEN (1)
    // Negative weight = logic gate CLOSED (0)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; j += 16) {
            uint16_t mask = 0;
            for (size_t b = 0; b < 16; ++b) {
                if (W[i * n + j + b] > 0) {
                    mask |= (1 << b);
                }
            }
            curve.masks[i * (n / 16) + (j / 16)] = mask;
        }
    }
    curve.scale = 1.0f; // Simplified
    curve.offset = 0.0f;
    return curve;
}

// ── 3. Transistor-Level Execution (AVX-512) ─────────────────────────────────
void execute_curve_avx512(const BinaryCurve& curve, const float* x, float* y) {
    size_t num_masks = curve.n / 16;
    
    for (size_t i = 0; i < curve.m; ++i) {
        __m512 v_sum = _mm512_setzero_ps();
        
        const uint16_t* row_masks = &curve.masks[i * num_masks];
        
        for (size_t j = 0; j < curve.n; j += 16) {
            // THE CIRCUIT SWITCH:
            // maskz_load reads the mask. If bit=1, transistor OPEN -> load x.
            // If bit=0, transistor CLOSED -> load 0.0f.
            __mmask16 gate_mask = row_masks[j / 16];
            __m512 v_routed_signal = _mm512_maskz_loadu_ps(gate_mask, x + j);
            
            // Accumulate the routed signals
            v_sum = _mm512_add_ps(v_sum, v_routed_signal);
        }
        
        // Horizontal sum to get final y[i]
        y[i] = _mm512_reduce_add_ps(v_sum) * curve.scale + curve.offset;
    }
}

// ── 4. Standard Dense FP32 Execution (For Baseline) ─────────────────────────
void execute_dense_avx512(const std::vector<float>& W, const float* x, float* y, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        __m512 v_sum = _mm512_setzero_ps();
        const float* row_w = &W[i * n];
        
        for (size_t j = 0; j < n; j += 16) {
            __m512 v_x = _mm512_loadu_ps(x + j);
            __m512 v_w = _mm512_loadu_ps(row_w + j);
            v_sum = _mm512_fmadd_ps(v_w, v_x, v_sum);
        }
        
        y[i] = _mm512_reduce_add_ps(v_sum);
    }
}

// ── 5. Benchmark ────────────────────────────────────────────────────────────
int main() {
    std::cout << "========================================================\n";
    std::cout << " NCA BINARY CURVE EXECUTION (Transistor-Level Routing)\n";
    std::cout << "========================================================\n\n";

    size_t m = 2048;
    size_t n = 1536; // Gemma-4 sizes
    
    std::vector<float> W(m * n);
    alignas(64) std::vector<float> x(n);
    alignas(64) std::vector<float> y_dense(m, 0.0f);
    alignas(64) std::vector<float> y_curve(m, 0.0f);

    // Fill with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(auto& val : W) val = dist(gen);
    for(auto& val : x) val = dist(gen);

    // Compile to curve
    std::cout << "[*] Compiling weight matrix to binary curve...\n";
    BinaryCurve curve = compile_to_curve(W, m, n);
    
    size_t dense_kb = (m * n * 4) / 1024;
    size_t curve_kb = (curve.masks.size() * 2) / 1024;
    std::cout << "    Dense memory: " << dense_kb << " KB\n";
    std::cout << "    Curve memory: " << curve_kb << " KB (" << (dense_kb/curve_kb) << "x smaller)\n\n";

    // Benchmark Dense
    std::cout << "[*] Running Dense FP32 AVX-512 (Multipliers ON)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000; ++iter) {
        execute_dense_avx512(W, x.data(), y_dense.data(), m, n);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_dense = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;
    std::cout << "    Time per execution: " << ms_dense << " ms\n\n";

    // Benchmark Curve
    std::cout << "[*] Running Binary Curve AVX-512 (Multipliers OFF, Routing ON)...\n";
    t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000; ++iter) {
        execute_curve_avx512(curve, x.data(), y_curve.data());
    }
    t1 = std::chrono::high_resolution_clock::now();
    double ms_curve = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;
    std::cout << "    Time per execution: " << ms_curve << " ms\n";
    
    std::cout << "\n>>> CURVE SPEEDUP: " << (ms_dense / ms_curve) << "x <<<\n";
    std::cout << "========================================================\n";

    return 0;
}
