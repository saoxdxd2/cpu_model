// ============================================================================
// NCA — Phase 4: GLR (Gated Linear RNN) Test
// tests/test_phase4_glr.cpp
//
// Tests the sequential backbone layer.
// Math: h_t = alpha * h_{t-1} + beta * x_t
// ============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <malloc.h>
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

// AVX-512 Phase 4 GLR Step
void glr_step_avx512(float* __restrict h, const float* __restrict alpha, const float* __restrict beta, const float* __restrict x, size_t d_size) {
    float* p_h = h;
    const float* p_a = alpha;
    const float* p_b = beta;
    const float* p_x = x;
    size_t rem = d_size;

    for (; rem >= 64; rem -= 64, p_h += 64, p_a += 64, p_b += 64, p_x += 64) [[likely]] {
        _mm_prefetch(reinterpret_cast<const char*>(p_h + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_a + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_b + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_x + 128), _MM_HINT_T0);

        // h = a * h + b * x
        __m512 v_bx0 = _mm512_mul_ps(_mm512_load_ps(p_b),      _mm512_load_ps(p_x));
        __m512 v_bx1 = _mm512_mul_ps(_mm512_load_ps(p_b + 16), _mm512_load_ps(p_x + 16));
        __m512 v_bx2 = _mm512_mul_ps(_mm512_load_ps(p_b + 32), _mm512_load_ps(p_x + 32));
        __m512 v_bx3 = _mm512_mul_ps(_mm512_load_ps(p_b + 48), _mm512_load_ps(p_x + 48));

        __m512 res0 = _mm512_fmadd_ps(_mm512_load_ps(p_a),      _mm512_load_ps(p_h),      v_bx0);
        __m512 res1 = _mm512_fmadd_ps(_mm512_load_ps(p_a + 16), _mm512_load_ps(p_h + 16), v_bx1);
        __m512 res2 = _mm512_fmadd_ps(_mm512_load_ps(p_a + 32), _mm512_load_ps(p_h + 32), v_bx2);
        __m512 res3 = _mm512_fmadd_ps(_mm512_load_ps(p_a + 48), _mm512_load_ps(p_h + 48), v_bx3);

        _mm512_store_ps(p_h,      res0);
        _mm512_store_ps(p_h + 16, res1);
        _mm512_store_ps(p_h + 32, res2);
        _mm512_store_ps(p_h + 48, res3);
    }
}

// AVX2 Phase 4 GLR Step
void glr_step_avx2(float* __restrict h, const float* __restrict alpha, const float* __restrict beta, const float* __restrict x, size_t d_size) {
    float* p_h = h;
    const float* p_a = alpha;
    const float* p_b = beta;
    const float* p_x = x;
    size_t rem = d_size;

    for (; rem >= 32; rem -= 32, p_h += 32, p_a += 32, p_b += 32, p_x += 32) [[likely]] {
        _mm_prefetch(reinterpret_cast<const char*>(p_h + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_a + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_b + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_x + 64), _MM_HINT_T0);

        __m256 v_bx0 = _mm256_mul_ps(_mm256_load_ps(p_b),      _mm256_load_ps(p_x));
        __m256 v_bx1 = _mm256_mul_ps(_mm256_load_ps(p_b + 8),  _mm256_load_ps(p_x + 8));
        __m256 v_bx2 = _mm256_mul_ps(_mm256_load_ps(p_b + 16), _mm256_load_ps(p_x + 16));
        __m256 v_bx3 = _mm256_mul_ps(_mm256_load_ps(p_b + 24), _mm256_load_ps(p_x + 24));

        __m256 res0 = _mm256_fmadd_ps(_mm256_load_ps(p_a),      _mm256_load_ps(p_h),      v_bx0);
        __m256 res1 = _mm256_fmadd_ps(_mm256_load_ps(p_a + 8),  _mm256_load_ps(p_h + 8),  v_bx1);
        __m256 res2 = _mm256_fmadd_ps(_mm256_load_ps(p_a + 16), _mm256_load_ps(p_h + 16), v_bx2);
        __m256 res3 = _mm256_fmadd_ps(_mm256_load_ps(p_a + 24), _mm256_load_ps(p_h + 24), v_bx3);

        _mm256_store_ps(p_h,      res0);
        _mm256_store_ps(p_h + 8,  res1);
        _mm256_store_ps(p_h + 16, res2);
        _mm256_store_ps(p_h + 24, res3);
    }
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  NCA — Phase 4: GLR Backbone Engine      ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    constexpr size_t D = 8192;
    constexpr size_t ITERS = 100000;

    // Allocate 64-byte aligned memory
    float* h = (float*)_aligned_malloc(D * sizeof(float), 64);
    float* a = (float*)_aligned_malloc(D * sizeof(float), 64);
    float* b = (float*)_aligned_malloc(D * sizeof(float), 64);
    float* x = (float*)_aligned_malloc(D * sizeof(float), 64);

    for(size_t i=0; i<D; i++) {
        h[i] = 1.0f; a[i] = 0.9f; b[i] = 0.1f; x[i] = 0.5f;
    }

    // Warmup
    for(size_t i=0; i<5000; i++) glr_step_avx512(h, a, b, x, D);

    uint64_t start_cycles = __rdtsc();
    auto t0 = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < ITERS; ++i) {
        glr_step_avx512(h, a, b, x, D);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t end_cycles = __rdtsc();
    
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ns_per_iter = (total_ms * 1e6) / ITERS;
    
    uint64_t total_cycles = end_cycles - start_cycles;
    double cycles_per_iter = static_cast<double>(total_cycles) / ITERS;
    double cycles_per_elem = cycles_per_iter / D;
    
    std::cout << "[Performance Profiling: GLR Step D=" << D << "]\n";
    std::cout << "  [AVX-512]\n";
    std::cout << "    Latency       : " << ns_per_iter << " ns / call\n";
    std::cout << "    CPU Cycles    : " << cycles_per_iter << " cycles / call\n";
    std::cout << "    Cycles / Elem : " << cycles_per_elem << " cycles / element\n\n";

    // AVX2 Benchmark
    for(size_t i=0; i<5000; i++) glr_step_avx2(h, a, b, x, D);
    start_cycles = __rdtsc();
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERS; ++i) glr_step_avx2(h, a, b, x, D);
    t1 = std::chrono::high_resolution_clock::now();
    end_cycles = __rdtsc();

    total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    ns_per_iter = (total_ms * 1e6) / ITERS;
    total_cycles = end_cycles - start_cycles;
    cycles_per_iter = static_cast<double>(total_cycles) / ITERS;
    cycles_per_elem = cycles_per_iter / D;

    std::cout << "  [AVX2]\n";
    std::cout << "    Latency       : " << ns_per_iter << " ns / call\n";
    std::cout << "    CPU Cycles    : " << cycles_per_iter << " cycles / call\n";
    std::cout << "    Cycles / Elem : " << cycles_per_elem << " cycles / element\n\n";

    if (cycles_per_elem < 0.5) {
        std::cout << "  Result: Operation is strictly Memory Bound. AVX-512 provides little to no compute advantage and may trigger frequency downclocking. AVX2 is the safer hybrid choice.\n\n";
    }

    _aligned_free(h);
    _aligned_free(a);
    _aligned_free(b);
    _aligned_free(x);

    return 0;
}
