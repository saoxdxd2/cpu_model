#pragma once
// ============================================================================
// NCA — MX Block-Quantized Linear Algebra
// core/linalg/mx_linear.hpp
//
// Extremely fast C-style raw memory structures for OCP MX quantization.
// Bypasses C++ std::vector overhead.
// ============================================================================

#include <cstdint>
#include <cstddef>

namespace nca::linalg {

// C-style pure data struct for weights (Signed INT8)
struct MXINT8Tensor {
    int8_t* __restrict data;
    uint8_t* __restrict scales;
    size_t num_blocks; // Number of 32-element blocks
};

// C-style pure data struct for activations (Unsigned UINT8 for VNNI compliance)
struct MXUINT8Tensor {
    uint8_t* __restrict data;
    uint8_t* __restrict scales;
    size_t num_blocks;
};

// C-style allocators guaranteeing 64-byte alignment
void mx_alloc_int8(MXINT8Tensor* t, size_t num_blocks);
void mx_free_int8(MXINT8Tensor* t);

void mx_alloc_uint8(MXUINT8Tensor* t, size_t num_blocks);
void mx_free_uint8(MXUINT8Tensor* t);

// Quantization (from float32)
void mx_quantize_w(const float* __restrict in, MXINT8Tensor* __restrict out);
void mx_quantize_x(const float* __restrict in, MXUINT8Tensor* __restrict out);

// Dynamically Dispatched Dot Product
float mx_dot(const MXINT8Tensor* __restrict w, const MXUINT8Tensor* __restrict x);

} // namespace nca::linalg
