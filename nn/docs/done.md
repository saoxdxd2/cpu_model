# NCA — Detailed Completed Tasks & Changelog

## 1. Research, Architecture, and Deep Math Specification (Completed)
- **Analyzed Hardware Targets**: Locked in Intel Core i5-1035G1 (Ice Lake), focusing specifically on maximizing the single AVX-512 unit (port 0) for VNNI capabilities while relying on AVX2 for baseline parallel throughput. Mapped out the 3-5 GB RAM ceiling.
- **Architected Multi-Agent Latent System**: Devised the 3-Core System (Vision, Logic, Action) to solve the parameter-bloat issue.
  - Adopted Zero-Copy Latent Tensors (`torch::Tensor` pointers) to bypass Tokenizer/Detokenizer serialization overheads entirely.
  - Specified "Latent Adapters" to linearly project representations between the cores.
- **Specified E-AdaPrune & AIR for Vision**: 
  - Abandoned fixed ViT grids (196+ tokens). 
  - Substituted with SVD-based spectral energy budget (1 to 64 tokens, $\tau=0.95$).
  - Integrated AIR (Optimal Transport via Sinkhorn) for dynamic, human-like saccadic region selection.
- **Specified Triple-Hybrid Logic Core**:
  - Combined GLR (Gated Linear Recurrence, 65%) for O(1) baseline reasoning.
  - Combined Selective SSM (20%) for long-range dynamics.
  - Combined Sparse Local Attention (SLA, W=256, 15%) for exact copying and matching.
- **Designed ACT (Adaptive Computation Time) & System 2 Thinking**:
  - Implemented the concept of L-Mode and H-Mode cycles using a Layer-23 Halting Gate based on output entropy.
  - Defined the `<think>` and `<answer>` token routing to use the 4096-token sequence as a scratchpad.
- **Locked Custom Number Formats (OCP MX)**:
  - Eliminated purely float/logarithmic formats in favor of VNNI-compatible block quantization.
  - **Logic & Vision Weights**: `MXINT8` (block of 32 INT8 + 1 E8M0 scale byte).
  - **Action Weights**: `MXINT4` (packed 4-bit + 1 E8M0 scale byte).
  - **Activations**: Dynamic per-token `INT8` directly targeting the `VPDPBUSD` instruction.
  - **Recurrent State**: Maintained at `BF16`/`FP32` to guarantee complex reasoning paths (like math proofs) don't accumulate rounding errors during multi-cycle loops.
- **Incorporated LeSTD**: Defined the post-training ultra-sparse Tucker Decomposition strategy to shrink weight tensors by 5-10x natively on CPU.
- **Defined Training & Distillation Strategy**:
  - Confirmed **English-only** vocabulary structure to compress the model.
  - Specified **KFAC (Second-Order Hessian)** optimization for rapid Colab free-tier convergence.
  - Specified Vision knowledge distillation heavily augmented by **YOLO v12/v18 object-centric priors**.
  - Confirmed hybrid C++/Python interop bridging C++ LibTorch inference and Python 3.14.5 Colab execution.

## 2. Phase 0: Project Foundation (Completed)
- **Directory Structure & Build System**: 
  - Initialized `CMakeLists.txt` at the root. 
  - Successfully overrode LibTorch's default `C++17` injection to strictly enforce **C++20** (`/std:c++20`, `/Zc:__cplusplus`).
  - Added strict MSVC compilation flags (`/W4`, `/permissive-`).
- **SIMD Runtime Dispatch Engine (`core/simd/dispatch.hpp` & `.cpp`)**:
  - Implemented `cpuid` based detection.
  - Safely probes for AVX2, FMA, AVX-512F, AVX-512BW, AVX-512VL, and the critical **AVX-512 VNNI**.
  - Wrote a static singleton cache so feature detection runs exactly once.
- **Global Config Single Source of Truth (`config/model_config.hpp`)**:
  - Defined `D_MODEL=2048`, `MAX_SEQ_LEN=4096`, `VOCAB_SIZE=32000`.
  - Defined `MX_BLOCK_SIZE=32`, mapping exactly to the OCP MX specification.
  - Laid out the GLR, SSM, and SLA intervals (SSM every 4 layers, SLA every 8 layers).
- **LibTorch Version Conflict Resolution**:
  - The initial environment had an ancient 2020 LibTorch (v1.4.0) causing `std::result_of` C++20 deprecation errors.
  - The developer manually upgraded the environment to **LibTorch 2.9.1+cpu**.
- **Phase 0 Test Execution (`tests/test_phase0.cpp`)**:
  - Triggered a full MSBuild compilation.
  - **RESULTS**: ALL CHECKS PASSED.
    - `C++20` standard macro `202002L` confirmed.
    - CPU detects Ice Lake extensions perfectly (`VNNI ready: YES`, `AVX2 ready: YES`).
    - LibTorch generated a test tensor with `torch::kCPU` without error.
    - MXINT8 byte sizing confirmed as 33 bytes per 32-weight block (8.25 bits/weight).
    - MXINT4 byte sizing confirmed as 17 bytes per 32-weight block (4.25 bits/weight).

## 3. Phase 1: AMI Normalization (Completed)
- **Implemented Atomic Math Unit**: Extracted `rmsnorm` computation into its own isolated `nca::math::rmsnorm` namespace and dispatcher.
- **Wrote AVX2 & AVX-512 Math Kernels**: 
  - Implemented the `_mm256_` logic inside `core/simd/avx2_kernels.cpp` explicitly built with `/arch:AVX2`.
  - Implemented the `_mm512_` logic inside `core/simd/avx512_kernels.cpp` explicitly built with `/arch:AVX512` to ensure safe isolated emissions in MSVC without bleeding into scalar logic.
- **Implemented Runtime Dispatch**: `core/normalization.cpp` evaluates the singleton CPU hardware capability dynamically and redirects the pointer immediately to the fastest kernel.
- **Phase 1 Test Execution (`tests/test_phase1_rmsnorm.cpp`)**:
  - Validated scalar, AVX2, and AVX-512 versions individually against a `torch::rsqrt` LibTorch reference pipeline. All kernels passed with near-zero error margins (< 5e-07).
  - Validated the fully integrated `nca::math::rmsnorm()` API which automatically engaged the `AVX-512` backend.
  - Reached ~9.6x native speedup on AVX-512 versus standard scalar normalization loops.

## 4. Phase 2: AMI Activations (Completed)
- **Implemented Generic SIMD Math Headers**: Created `core/simd/avx512_math.hpp` and `core/simd/avx2_math.hpp` containing inline, globally accessible mathematical approximations (e.g., `exp_ps`, `silu_ps`) for massive code reuse across specialized unrolled loops.
- **Branchless Math Operations**: Implemented a pure Minimax polynomial approximation for $2^{x \log_2 e}$ utilizing `_mm512_scalef_ps`, eliminating all `std::exp` branch delays.
- **Activation API**: Added `nca::math::silu()` mapping to `core/activations.cpp`.
- **Phase 2 Test Execution (`tests/test_phase2_activations.cpp`)**:
  - Achieved an 11.6x speedup over scalar `std::exp` loops.
  - Validated against `torch::silu` with error bounds < 5e-07.

## 5. Phase 3: The MX VNNI Math Core (Completed)
- **Proof-of-Concept Testing**: Validated `MXINT8` VNNI mathematics successfully.
- **C-Style Memory Framework**: Created `MXINT8Tensor` and `MXUINT8Tensor` C-structs mapping raw `_aligned_malloc` 64-byte bounded memory to strictly avoid C++ `std::vector` overhead.
- **VNNI Kernels**: Added `core/simd/avx512_vnni.cpp` featuring `_mm512_dpbusd_epi32` (VPDPBUSD) math to execute 64 MACs per cycle and reconstruct `E8M0` powers of two.
- **Vectorized Activation Quantization (`mx_quantize_x`)**: Completely resolved the massive 23-cycle scalar bottleneck. Built an AVX-512 implementation that vectorizes `max_abs` detection, scaling, and signed saturation via `_mm512_cvtepi32_epi8`. Brought quantization latency down to **0.45 cycles/element**.
- **Python-Driven Auto-Detector Suite**: Designed `autogenerate.py` hooked directly into `CMakeLists.txt`. It dynamically parses C++ headers, discovers functions, and generates a self-registering C++ benchmark binary `test_auto.exe`. This permanently eliminates manual test file maintenance.

## 6. Phase 4: GLR Block (Completed)
- **Gated Linear Recurrence**: Implemented `glr_step` inside `core/layers/glr.cpp` to execute the temporal memory pass: `h_t = alpha * h_{t-1} + beta * x_t`.
- **FMA Vectorization**: Used `_mm512_fmadd_ps` to compute the fused recurrence across 16 elements at once. 
- **CachePolicy Integration**: Applied `constexpr CachePolicy<4*D*4>` to select L2_STREAM mode at compile time — regular stores + prefetch distance 4 cache lines ahead.
- **Branchless Tail**: Replaced the scalar tail loop with `__mmask16` masked SIMD operations.
- **Performance Output**: Measured at **0.19 cycles/element**, memory-bandwidth limited.

## 7. Phase 5: SSM Block (Completed)
- **Selective State-Space Model**: Implemented `ssm_step` inside `core/layers/ssm.cpp` for Mamba-style continuous dynamics: `h = A*h + B*x, y = C*h`.
- **d_state=16 AVX-512 Fastpath**: When `d_state == 16`, the entire state vector maps to a single `__m512` register — the inner loop is fully vectorized with zero scalar code.
- **4x Unrolled**: Processes 4 channels per iteration with hoisted B/C broadcasts (stay in registers for the entire kernel).
- **CachePolicy**: Working set is 1.1MB → DDR4_NT policy selects `prefetch_dist=8` at compile time.
- **Performance Output**: Measured at **5.66 cycles/element**. The 1.1MB working set (h + A arrays) exceeds L2 cache; performance is DDR4 bandwidth-bound.

## 8. Phase 6: SLA Block (Completed)
- **Sparse Local Attention**: Implemented `sla_step` in `core/layers/sla.cpp` for sliding-window attention (W=256, d_head=128).
- **Online Softmax**: Fully branchless AVX-512 softmax with minimax exponential approximation.
- **CachePolicy**: Working set ~256KB fits L2, so `L2_STREAM` policy was automatically selected via `constexpr CachePolicy`.
- **Performance Output**: Achieved **~0.88 cycles/element**.

## 9. Phase 7: ACT & MLP Block (Completed)
- **Gated MLP**: Implemented `gated_mlp_step` inside `core/layers/mlp.cpp` incorporating a fused `SwiGLU -> Quantize` pass to avoid intermediate memory spills to DDR4.
- **Halting Gate**: Implemented `halting_step` in `core/layers/halting.cpp` using `mx_dot` to compute the halting probability for Adaptive Computation Time (ACT).
- **Performance Output**: Achieved **~0.72 cycles/elem** for fused SwiGLU quantize, **~2.4 cycles/elem** for full MLP block, and **~0.50 cycles/elem** for the Halting classifier.

## 10. Phase 8: Vision Stage 1 (Completed)
- **Global Scanner**: Implemented `dwconv2d_3x3` for local feature extraction and `ssm2d_scan` for global receptive field in `core/vision/scanner.cpp`.
- **NHWC Layout**: Processed depthwise convolutions using the NHWC (channels last) memory layout, allowing AVX-512 to naturally vectorize over the `C` dimension. Fully unrolled the 3x3 kernel into 9 fused multiply-adds (`_mm512_fmadd_ps`).
- **Performance Output**: Achieved **~0.68 cycles/element** for the 3x3 DW-Conv, and **~6.2 cycles/element** for the 2D-SSM scan (matching the highly optimized 1D SSM backbone).

## 11. SSM Layer Fusion Optimization (Completed)
- **Fused Kernel**: Implemented `mx_fused_ssm_silu_quantize_step` in `core/layers/ssm.cpp`.
- **Memory Footprint**: By fusing the operations and locally buffering `y` chunks inside the AVX-512 loop, we completely bypassed L2/L3 global memory writes for the intermediate floating-point state, reducing memory traffic by 64KB per inference step.
- **Performance Output**: Measured at **~5.47 cycles/element**. While micro-benchmarks show parity with discrete calls (due to L1 hot-cache hits), the fusion massively cuts DDR4 bandwidth contention in the full network pipeline.
