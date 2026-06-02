#pragma once
// ============================================================================
// CENTAUR — Graph Executor (Runtime Walker)
// centaur/graph_executor.hpp
//
// Walks the compiled ExecutionGraph with ZERO dynamic decisions.
// Every branch was resolved at compile time. Every prefetch is baked in.
// The executor is a flat loop over cache-line-aligned ExecNodes.
//
// Features CCSE Batched Expert GEMM support.
// ============================================================================

#include "core/centaur/execution_graph.hpp"
#include <immintrin.h>
#include <cstring>
#include <cmath>

namespace nca::centaur {

class GraphExecutor {
public:
    explicit GraphExecutor(const ExecutionGraph& graph)
        : graph_(graph)
    {
        // Allocate the unified arena (64-byte aligned)
        arena_size_ = graph_.arena_size_bytes;
        if (arena_size_ > 0) {
#ifdef _WIN32
            arena_ = static_cast<uint8_t*>(_aligned_malloc(arena_size_, 64));
#else
            arena_ = static_cast<uint8_t*>(aligned_alloc(64, arena_size_));
#endif
            std::memset(arena_, 0, arena_size_);
        }
    }

    ~GraphExecutor() {
        if (arena_) {
#ifdef _WIN32
            _aligned_free(arena_);
#else
            free(arena_);
#endif
        }
    }

    GraphExecutor(const GraphExecutor&) = delete;
    GraphExecutor& operator=(const GraphExecutor&) = delete;
    GraphExecutor(GraphExecutor&& o) noexcept
        : graph_(o.graph_), arena_(o.arena_), arena_size_(o.arena_size_) {
        o.arena_ = nullptr; o.arena_size_ = 0;
    }

    // ── Slot accessor ───────────────────────────────────────────────────
    float* slot_ptr(uint16_t slot_id) {
        if (slot_id >= graph_.slots.size()) return nullptr;
        return reinterpret_cast<float*>(arena_ + graph_.slots[slot_id].offset_bytes);
    }

    const float* slot_ptr(uint16_t slot_id) const {
        if (slot_id >= graph_.slots.size()) return nullptr;
        return reinterpret_cast<const float*>(arena_ + graph_.slots[slot_id].offset_bytes);
    }

    // ── Execute the full graph ──────────────────────────────────────────
    void execute(size_t d_model, size_t batch_size = 1) {
        const ExecNode* __restrict nodes = graph_.nodes.data();
        const size_t n = graph_.nodes.size();

        for (size_t i = 0; i < n; ++i) [[likely]] {
            const ExecNode& node = nodes[i];

            switch (node.type) {
                case NodeType::PREFETCH:
                    execute_prefetch(node, batch_size);
                    break;
                case NodeType::GLR_RECURRENCE:
                    execute_glr(node, d_model, batch_size);
                    break;
                case NodeType::SPECTRAL_FWHT:
                    execute_fwht(node, d_model, batch_size);
                    break;
                case NodeType::SPECTRAL_IFWHT:
                    execute_ifwht(node, d_model, batch_size);
                    break;
                case NodeType::UNIFIED_BCT:
                    execute_unified_bct(node, d_model, batch_size);
                    break;
                case NodeType::RMSNORM:
                    execute_rmsnorm(node, d_model, batch_size);
                    break;
                case NodeType::ACTIVATION:
                    execute_activation(node, d_model, batch_size);
                    break;
                case NodeType::RESIDUAL_ADD:
                    execute_residual(node, d_model, batch_size);
                    break;
                case NodeType::FENCE:
                    _mm_sfence();
                    break;
            }
        }
    }

    // ── Load state into the primary slot ────────────────────────────────
    void load_state(const float* src, size_t d_model, size_t batch_size = 1) {
        float* dst = slot_ptr(0); 
        if (dst && src) std::memcpy(dst, src, d_model * batch_size * sizeof(float));
    }

    // ── Extract state from the primary slot ─────────────────────────────
    void extract_state(float* dst, size_t d_model, size_t batch_size = 1) const {
        const float* src = slot_ptr(0);
        if (dst && src) std::memcpy(dst, src, d_model * batch_size * sizeof(float));
    }

    // ── Load weights into their respective slots ────────────────────────
    void load_weights(uint16_t slot_id, const float* weights, size_t count) {
        float* dst = slot_ptr(slot_id);
        if (dst && weights) std::memcpy(dst, weights, count * sizeof(float));
    }

    size_t arena_size() const { return arena_size_; }

private:
    const ExecutionGraph& graph_;
    uint8_t* arena_ = nullptr;
    size_t   arena_size_ = 0;
    RouteResult route_results_[32]{};

    // ── Node executors ──────────────────────────────────────────────────

    void execute_prefetch(const ExecNode& node, size_t batch_size) {
        if (node.prefetch_hint == 3) {
            // UNIFIED BCT dynamic expert prefetch could go here
            // (omitted for now since BCT experts are L2-pinned anyway)
            return;
        }

        uint16_t target = node.prefetch_target_slot;
        if (target >= graph_.slots.size()) return;
        const auto& slot = graph_.slots[target];
        const char* base = reinterpret_cast<const char*>(arena_ + slot.offset_bytes);
        size_t lines = slot.size_bytes / 64;
        size_t depth = (std::min)(lines, graph_.plan.prefetch.depth);

        if (node.prefetch_hint == 2) {
            for (size_t j = 0; j < depth; ++j) _mm_prefetch(base + j * 64, _MM_HINT_NTA);
        } else if (node.prefetch_hint == 1) {
            for (size_t j = 0; j < depth; ++j) _mm_prefetch(base + j * 64, _MM_HINT_T1);
        } else {
            for (size_t j = 0; j < depth; ++j) _mm_prefetch(base + j * 64, _MM_HINT_T0);
        }
    }

    void execute_glr(const ExecNode& node, size_t D, size_t batch_size) {
        float* state_base = slot_ptr(node.input_slot);
        const float* alpha = slot_ptr(node.weight_slot);
        const float* beta  = slot_ptr(node.aux_slot);
        if (!state_base || !alpha || !beta) return;

        for (size_t b = 0; b < batch_size; ++b) {
            float* state = state_base + b * D;
            size_t rem = D;
            float* p_s = state;
            const float* p_a = alpha;
            const float* p_b = beta;

#if defined(__AVX512F__)
            for (; rem >= 64; rem -= 64, p_s += 64, p_a += 64, p_b += 64) [[likely]] {
                _mm_prefetch(reinterpret_cast<const char*>(p_a + 128), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(p_b + 128), _MM_HINT_T0);
                for (int k = 0; k < 4; ++k) {
                    __m512 vs = _mm512_loadu_ps(p_s + k * 16);
                    __m512 va = _mm512_loadu_ps(p_a + k * 16);
                    __m512 vb = _mm512_loadu_ps(p_b + k * 16);
                    _mm512_storeu_ps(p_s + k * 16, _mm512_fmadd_ps(va, vs, vb));
                }
            }
            for (; rem >= 16; rem -= 16, p_s += 16, p_a += 16, p_b += 16) {
                __m512 vs = _mm512_loadu_ps(p_s);
                _mm512_storeu_ps(p_s, _mm512_fmadd_ps(_mm512_loadu_ps(p_a), vs, _mm512_loadu_ps(p_b)));
            }
#elif defined(__AVX2__)
            for (; rem >= 32; rem -= 32, p_s += 32, p_a += 32, p_b += 32) [[likely]] {
                for (int k = 0; k < 4; ++k) {
                    __m256 vs = _mm256_loadu_ps(p_s + k * 8);
                    __m256 va = _mm256_loadu_ps(p_a + k * 8);
                    __m256 vb = _mm256_loadu_ps(p_b + k * 8);
                    _mm256_storeu_ps(p_s + k * 8, _mm256_fmadd_ps(va, vs, vb));
                }
            }
#endif
            for (size_t j = 0; j < rem; ++j) {
                p_s[j] = p_a[j] * p_s[j] + p_b[j];
            }
        }
    }

    void execute_fwht(const ExecNode& node, size_t D, size_t batch_size) {
        float* data_base = slot_ptr(node.input_slot);
        if (!data_base) return;

        for (size_t b = 0; b < batch_size; ++b) {
            float* data = data_base + b * D;
            for (size_t len = 1; len < D; len <<= 1) {
                for (size_t i = 0; i < D; i += len << 1) {
                    for (size_t j = 0; j < len; ++j) {
                        float u = data[i + j];
                        float v = data[i + j + len];
                        data[i + j]       = u + v;
                        data[i + j + len] = u - v;
                    }
                }
            }
        }
    }

    void execute_ifwht(const ExecNode& node, size_t D, size_t batch_size) {
        execute_fwht(node, D, batch_size); 
    }

    void execute_unified_bct(const ExecNode& node, size_t D, size_t batch_size) {
        float* state = slot_ptr(node.input_slot);
        if (!state || !graph_.bct_engine) return;
        
        for (size_t b = 0; b < batch_size; ++b) {
            graph_.bct_engine->execute(state + b * D, state + b * D);
        }
    }

    void execute_rmsnorm(const ExecNode& node, size_t D, size_t batch_size) {
        float* data_base = slot_ptr(node.input_slot);
        if (!data_base) return;

        for (size_t b = 0; b < batch_size; ++b) {
            float* data = data_base + b * D;
            float sum_sq = 0.0f;
#if defined(__AVX512F__)
            __m512 v_acc = _mm512_setzero_ps();
            size_t rem = D;
            const float* p = data;
            for (; rem >= 16; rem -= 16, p += 16) {
                __m512 v = _mm512_loadu_ps(p);
                v_acc = _mm512_fmadd_ps(v, v, v_acc);
            }
            sum_sq = _mm512_reduce_add_ps(v_acc);
            for (size_t j = 0; j < rem; ++j) sum_sq += p[j] * p[j];
#else
            for (size_t j = 0; j < D; ++j) sum_sq += data[j] * data[j];
#endif
            float rms = 1.0f / std::sqrt(sum_sq / D + 1e-6f);

#if defined(__AVX512F__)
            __m512 v_rms = _mm512_set1_ps(rms);
            rem = D; float* q = data;
            for (; rem >= 16; rem -= 16, q += 16) {
                _mm512_storeu_ps(q, _mm512_mul_ps(_mm512_loadu_ps(q), v_rms));
            }
            for (size_t j = 0; j < rem; ++j) q[j] *= rms;
#else
            for (size_t j = 0; j < D; ++j) data[j] *= rms;
#endif
        }
    }

    void execute_activation(const ExecNode& node, size_t D, size_t batch_size) {
        float* data_base = slot_ptr(node.input_slot);
        if (!data_base) return;
        for (size_t b = 0; b < batch_size; ++b) {
            float* data = data_base + b * D;
            for (size_t j = 0; j < D; ++j)
                data[j] = data[j] / (1.0f + std::exp(-data[j]));
        }
    }

    void execute_residual(const ExecNode& node, size_t D, size_t batch_size) {
        float* out_base = slot_ptr(node.output_slot);
        const float* in_base = slot_ptr(node.input_slot);
        if (!out_base || !in_base) return;
        
        for (size_t b = 0; b < batch_size; ++b) {
            float* out = out_base + b * D;
            const float* in = in_base + b * D;
#if defined(__AVX512F__)
            size_t rem = D;
            for (; rem >= 16; rem -= 16, out += 16, in += 16)
                _mm512_storeu_ps(out, _mm512_add_ps(_mm512_loadu_ps(out), _mm512_loadu_ps(in)));
            for (size_t j = 0; j < rem; ++j) out[j] += in[j];
#else
            for (size_t j = 0; j < D; ++j) out[j] += in[j];
#endif
        }
    }
};

} // namespace nca::centaur
