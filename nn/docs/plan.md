# NCA — Nano-Core Architecture Implementation Plan

> **Status**: Comprehensive Master Blueprint (v2.2)
> **Goal**: Beat Transformers on CPU inference for multimodal tasks (Text + Image -> Text).

## 1. Project Philosophy & Hardware Targets

### 1.1 Development Methodology & Language Interop
- **Hybrid Codebase**: Complex ASM/C/C++ for inference performance, paired with Python 3.14.5 for the Colab training orchestration.
- **Atomic Mathematical Isolation (AMI)**: Each mathematical operation is isolated into its own atomic, testable unit. Once optimized and proven correct, it is deeply documented and *never touched again* unless bottlenecked.
- **Test-First Paradigm**: We test every concept in a dedicated test file first. We apply it to the main codebase only after validation. We clean up test artifacts after completion. 
- **Enterprise-Level C++**: Code must be explicit, memory-safe, strictly adhering to **C++20**. No "AI spaghetti code" that developers can't understand.

### 1.2 Hardware Environment & Constraints
- **Target Inference Hardware**: 500 units of HP 250 G8 Notebook PC.
- **CPU**: Intel Core i5-1035G1 (Ice Lake) @ 1.00GHz base / 3.6GHz turbo. 4 Cores / 8 Threads.
- **Memory Constraint**: 3 to 5 GB RAM budget for the model (out of 8 GB total RAM).
- **SIMD Capabilities**: AVX2 (constant baseline, 2 ports) + AVX-512 (1 unit boost, port 0). Crucially, includes **VNNI (`VPDPBUSD`)**, VBMI2, BITALG, and VPOPCNTDQ.
- **Training Environment**: Google Colab Free Tier (GPU T4 or TPU v2) using Python 3.14.5.
- **Language Scope**: strictly **English-only** to heavily compress the embedding matrix and vocabulary size.

## 2. The Multi-Agent Nano-Core Architecture

Instead of a single monolithic model with an attention bottleneck, NCA splits intelligence into three specialized "Nano-Cores" that communicate natively in the latent space.

### 2.1 Latent Communication (Zero-Copy Inter-Agent Protocol)
- **Bypassing the Tokenizer**: Standard AutoGPT setups destroy CPU performance because Agent A tokenizes output to text, passes a string, and Agent B detokenizes it. We skip this entirely.
- **Raw Tensor Passing**: Output matrix of one core stays physically resident in the i5-1035G1's L3 Cache/RAM (`torch::Tensor`). We pass the memory pointer directly to the next core.
- **Latent Adapters**: To bridge different internal representations between cores, we train a tiny, single-layer linear projection (Adapter `V->L` and `L->A`). These cost microseconds on AVX-512 VNNI.

### 2.2 The Three Nano-Cores

**Core 1: Vision Core (~0.8 GB, ~400M params)**
- **Role**: Extract deep visual features. Does *not* output text. Outputs a latent tensor `[Batch, K, 2048]`.
- **YOLO-Inspired Distillation Priors**: Training leverages knowledge distillation from a pretrained vision encoder, heavily enhanced with object-centric priors (borrowing ideas from YOLO v12 / v18 architectures) to ensure high intelligence density in a CPU-compatible footprint.
- **E-AdaPrune Spectral Budget**: Instead of fixed patches (like ViT's 196), it treats token budget as an intrinsic property of the image's information density.
  - SVD (Singular Value Decomposition) analyzes the feature matrix. 
  - A blank wall allocates 1-4 tokens. A complex textbook allocates 50-64 tokens. Threshold $\tau = 0.95$.
- **AIR (Optimal Transport)**: Uses Optimal Transport (Sinkhorn) to select only patches most aligned with the current task, suppressing redundancy and mimicking human saccadic attention.
- **Pipeline Stages**:
  1. *Global Scanner*: DW-Conv + 2D-SSM for global context on a 384x384 image down to 24x24 features.
  2. *Region Proposer*: Adaptive Top-K based on spectral budget.
  3. *Detail Inspector*: Per-region SSM processing.
  4. *Semantic Compressor*: Cross-Attention squeeze into $K$ visual tokens.

**Core 2: Logic Core (~1.5 GB, ~800M params)**
- **Role**: The "Brain". Handles heavy reasoning, dense linear algebra, KFAC optimizations, and knowledge retrieval.
- **Triple-Hybrid Recurrence Backbone (24 Layers)**:
  - **65% Gated Linear Recurrence (GLR)**: Fused linear attention + recurrence (O(1) per token).
  - **20% Selective SSM (Mamba-style)**: For long-range continuous dependency tracking.
  - **15% Sparse Local Attention (SLA)**: Windowed attention (W=256) for exact pattern recall.
- **ACT (Adaptive Computation Time) / L-H Cycles**:
  - A *Halting Gate* at Layer 23 evaluates output state entropy.
  - *L-Mode (Low Cycle)*: Confident logic -> immediate exit (1x compute).
  - *H-Mode (High Cycle)*: Uncertain logic -> loops output state back to Layer 0 (up to 4 cycles). Acts dynamically as 24, 48, 72, or 96 layers based on task difficulty.
- **System 2 "Latent Thinking"**: The model outputs a `<think>` token to use its 4096-token sequence buffer as a scratchpad. No text is emitted to the user until the `<answer>` token is generated.

**Core 3: Action Core (~1.0 GB, ~500M params)**
- **Role**: Specialized for high-speed text/code generation.
- **Backbone**: 16 Layers of pure GLR.
- Takes the fully reasoned latent tensor from the Logic Core and converts it directly into fluent English or C++ code. Always 1-cycle.

## 3. Advanced Mathematics & Compression

### 3.1 OCP Microscaling (MX) Block Quantization
- **Research Basis**: Per-tensor INT4 collapses logic, but grouping weights into blocks of 32 with a shared E8M0 scale completely restores accuracy (May 2026 Research).
- **Data Type Mapping**:
  - **Weights (Vision & Logic)**: **MXINT8** (8-bit int + E8M0 power-of-2 scale per block of 32).
  - **Weights (Action)**: **MXINT4** (Packed 4-bit int + E8M0 scale per block of 32) to half memory bandwidth.
  - **Activations**: **Dynamic INT8**. VNNI `VPDPBUSD` instruction *requires* 8-bit integers. We dynamically quantize activations per-token.
  - **Recurrent States & Latent Communication**: **BF16 / FP32**. Never quantized. State corruption from rounding errors during "High Cycle" loops must be avoided at all costs.

### 3.2 LeSTD (Learning-based Sparse Tensor Decomposition)
- **Math**: Replaces standard SVD with global Tucker decomposition. 
  $$\mathcal{W} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C$$
- **Implementation**: Learns a shared low-dimensional basis across multiple attention heads and isolates cross-mode interactions into an ultra-sparse core tensor $\mathcal{G}$.
- **CPU Advantage**: Applying a grounded pruning mechanism to $\mathcal{G}$ yields highly compact sparse matrices, routed directly through the AVX2/AVX-512 hybrid dispatcher.

### 3.3 Training Strategy: KFAC & Hessian Second-Order Optimization
- To rapidly train this novel architecture on the Colab free tier, we avoid standard Adam/SGD limitations.
- **KFAC (Kronecker-factored Approximate Curvature)**: Uses second-order Hessian approximations to dramatically accelerate convergence.
- **Adapter Backprop Game**: During collaborative training, the frozen cores communicate via Latent Adapters, and only the Adapters are optimized via backpropagation, keeping VRAM usage well within free-tier GPU/TPU limits.

## 4. Step-by-Step Implementation Phasing

| Phase | Subsystem | Description | Status |
|---|---|---|---|
| **0** | **Foundation** | CMake C++20, LibTorch 2.9.1+cpu setup, CPUID SIMD detection, config constants, MX block size verification tests. | **COMPLETED** |
| **1** | **AMI Normalization** | FP32 / BF16 RMSNorm kernels. Strict high-precision for recurrent states. AVX2/AVX-512 paths. | **COMPLETED** |
| **2** | **AMI Activations** | GELU and SiLU kernels. Tested and isolated. | **COMPLETED** |
| **3** | **AMI MX Math Core** | VNNI matrix multiplications. MXINT8/MXINT4 block unpacking, INT8 dynamic activation generation, `VPDPBUSD` execution, E8M0 scale application. | **COMPLETED** |
| **4** | **GLR Block** | Gated Linear Recurrence state tracking, diagonal gates, fused linear attention. | **COMPLETED** |
| **5** | **SSM Block** | Selective State-Space Model continuous dynamics scan. | **COMPLETED** |
| **6** | **SLA Block** | Sparse Local Attention sliding window masking (W=256). | **COMPLETED** |
| **7** | **ACT & MLP** | GatedMLP implementation and the Halting Gate (Entropy classifier + L/H cycling logic). | **COMPLETED** |
| **8** | **Vision: Stage 1** | Global Scanner (DW-Conv + 2D-SSM). | **COMPLETED** |
| **9** | **Vision: Stage 2** | E-AdaPrune Spectral Budget (SVD to determine $K$ tokens). | Pending |
| **10** | **Vision: Stage 3** | Region Proposer & AIR Optimal Transport (Sinkhorn patch selection). | Pending |
| **11** | **Vision: Stage 4** | Detail Inspector & Semantic Compressor (Cross-Attention squeeze). | Pending |
| **12** | **Latent Adapters** | FP32 Micro-linear projections mapping 2048D spaces between cores. | Pending |
| **13** | **Vision Core** | Assembly of Vision Core (with YOLO distillation logic hooks). | Pending |
| **14** | **Logic Core** | Assembly of Logic Core + Think Loop logic. | Pending |
| **15** | **Action Core** | Assembly of Action Core. | Pending |
| **16** | **Inference Engine** | Final C++ token generation loop bridging all cores. | Pending |
| **17** | **Python Interop** | Python 3.14.5 bindings to bridge C++ inference with PyTorch/Colab training loop. | Pending |
| **18** | **KFAC Optimizer** | Custom C++ / PyTorch hybrid KFAC optimizer for second-order Latent Adapter training. | Pending |
