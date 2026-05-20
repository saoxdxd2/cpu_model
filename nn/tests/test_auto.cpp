// tests/test_auto.cpp
#include "core/activations.hpp"
#include "core/normalization.hpp"
#include "core/linalg/mx_linear.hpp"
#include "core/layers/glr.hpp"
#include "core/simd/dispatch.hpp"

#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <chrono>
#include <malloc.h>
#include <tuple>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

// ── 1. Profiler Core ──────────────────────────────────────────────
struct Profiler {
    size_t D;
    size_t iters = 50000;
    double ns_per_call = 0;
    double cycles_per_elem = 0;

    template<typename F>
    void run(F&& func) {
        for(size_t i=0; i<1000; i++) func(); // Warmup
        
        uint64_t c0 = __rdtsc();
        auto t0 = std::chrono::high_resolution_clock::now();
        for(size_t i=0; i<iters; i++) func();
        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t c1 = __rdtsc();

        ns_per_call = (std::chrono::duration<double, std::milli>(t1 - t0).count() * 1e6) / iters;
        cycles_per_elem = static_cast<double>(c1 - c0) / iters / D;
    }
};

// ── 2. Automatic Argument Generation via Template Metaprogramming ──
template <typename T> struct ArgGen;

template <> struct ArgGen<std::span<float>> {
    float* ptr; size_t d;
    ArgGen(size_t D) : d(D) { ptr = (float*)_aligned_malloc(D*sizeof(float), 64); }
    ~ArgGen() { _aligned_free(ptr); }
    std::span<float> get() { return std::span<float>(ptr, d); }
};

template <> struct ArgGen<std::span<const float>> {
    float* ptr; size_t d;
    ArgGen(size_t D) : d(D) { ptr = (float*)_aligned_malloc(D*sizeof(float), 64); }
    ~ArgGen() { _aligned_free(ptr); }
    std::span<const float> get() { return std::span<const float>(ptr, d); }
};

template <> struct ArgGen<float*> {
    float* ptr;
    ArgGen(size_t D) { ptr = (float*)_aligned_malloc(D*sizeof(float), 64); }
    ~ArgGen() { _aligned_free(ptr); }
    float* get() { return ptr; }
};

template <> struct ArgGen<const float*> {
    float* ptr;
    ArgGen(size_t D) { ptr = (float*)_aligned_malloc(D*sizeof(float), 64); }
    ~ArgGen() { _aligned_free(ptr); }
    const float* get() { return ptr; }
};

template <> struct ArgGen<const nca::linalg::MXINT8Tensor&> {
    nca::linalg::MXINT8Tensor tensor;
    ArgGen(size_t D) : tensor(D/32) {}
    const nca::linalg::MXINT8Tensor& get() { return tensor; }
};

template <> struct ArgGen<const nca::linalg::MXUINT8Tensor&> {
    nca::linalg::MXUINT8Tensor tensor;
    ArgGen(size_t D) : tensor(D/32) {}
    const nca::linalg::MXUINT8Tensor& get() { return tensor; }
};

template <> struct ArgGen<nca::linalg::MXUINT8Tensor&> {
    nca::linalg::MXUINT8Tensor tensor;
    ArgGen(size_t D) : tensor(D/32) {}
    nca::linalg::MXUINT8Tensor& get() { return tensor; }
};

template <> struct ArgGen<size_t> {
    size_t d;
    ArgGen(size_t D) : d(D) {}
    size_t get() { return d; }
};

// ── 3. Framework Runner ───────────────────────────────────────────
template<typename... Args>
void run_auto_benchmark(const char* name, void(*func)(Args...), size_t D) {
    Profiler p; p.D = D;
    std::tuple<ArgGen<std::decay_t<Args>>...> args{ArgGen<std::decay_t<Args>>(D)...};
    
    p.run([&]() {
        std::apply([&](auto&... a) { func(a.get()...); }, args);
    });
    
    std::cout << "  [" << name << "]\n";
    std::cout << "    Latency       : " << p.ns_per_call << " ns / call\n";
    std::cout << "    Cycles / Elem : " << p.cycles_per_elem << " cycles / element\n\n";
}

// ── 4. Global Registry ────────────────────────────────────────────
struct Registry {
    using TestFunc = std::function<void(size_t)>;
    struct Entry { std::string name; TestFunc func; };
    static std::vector<Entry>& get() { static std::vector<Entry> inst; return inst; }
    static void add(const char* name, TestFunc f) { get().push_back({name, f}); }
};

// Macro automatically expands the function pointer and type deduction
#define AUTO_PROFILE(Func) \
    struct _AutoReg_##Func { \
        _AutoReg_##Func() { \
            Registry::add(#Func, [](size_t D){ run_auto_benchmark(#Func, &Func, D); }); \
        } \
    } _reg_obj_##Func;

// ============================================================================
// SINGLE LINE BENCHMARK REGISTRATION
// ============================================================================

AUTO_PROFILE(nca::math::silu);
AUTO_PROFILE(nca::math::rmsnorm);
AUTO_PROFILE(nca::linalg::mx_dot);
AUTO_PROFILE(nca::linalg::mx_quantize_x);
AUTO_PROFILE(nca::layers::glr_step);

// ============================================================================

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  NCA — Template-Deducing Profiler        ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";
    
    std::cout << nca::simd::detect().to_string() << "\n";
    std::cout << "Active Backend: " << nca::simd::backend_name(nca::simd::best_backend()) << "\n\n";
    
    size_t D = 8192;
    std::cout << "[Profiling Vector Length D=" << D << "]\n\n";

    for (const auto& entry : Registry::get()) {
        entry.func(D);
    }
    
    return 0;
}
