#pragma once
// ============================================================================
// NCA -- Vision Stage 1: Global Scanner (Phase 8)
// core/vision/scanner.hpp
//
// Depthwise Convolution (DW-Conv) and 2D-SSM for vision encoding.
// Expects input in NHWC (channels last) format to maximize AVX-512 ILP.
// ============================================================================

#include <cstddef>

namespace nca::vision {

struct ScannerConfig {
    size_t H = 16;
    size_t W = 16;
    size_t C = 128;
};

// 3x3 Depthwise Convolution
// input:  [H, W, C]
// weight: [3, 3, C]
// output: [H, W, C]
void dwconv2d_3x3(
    const float* __restrict input,
    const float* __restrict weight,
    float* __restrict output,
    ScannerConfig cfg
);

// 2D-SSM Scan (Flattens H*W into a 1D sequence and applies SSM)
// h: [H*W, d_state]
// A: [H*W, d_state]
// B: [d_state]
// C_proj: [d_state]
// x: [H*W, C]
// y: [H*W, C]
void ssm2d_scan(
    float* __restrict h,
    const float* __restrict A,
    const float* __restrict B,
    const float* __restrict C_proj,
    const float* __restrict x,
    float* __restrict y,
    ScannerConfig cfg
);

} // namespace nca::vision
