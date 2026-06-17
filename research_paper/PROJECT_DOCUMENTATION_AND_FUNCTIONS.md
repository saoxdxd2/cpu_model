# CENTAUR Neural Engine: Exhaustive Codebase Documentation (LangChain AI Generated)

This document was generated concurrently using LangChain and Gemini 1.5 Pro for precise, deduplicated architectural insights.

---

## File: `CMakeLists.txt`

### Architectural Analysis: `CMakeLists.txt` (CENTAUR Neural Engine)

This configuration file serves as the foundational build-orchestrator for the CENTAUR engine, enforcing strict hardware-level constraints required for high-throughput, low-latency neural inference.

#### 1. Core Purpose
The script enforces a **static-first, hardware-locked build environment**. By mandating C++20 and MSVC-specific AVX-512 instruction sets, it ensures that the compiler generates dense, SIMD-vectorized machine code, eliminating the need for runtime dispatching or dynamic polymorphism that would otherwise introduce branch misprediction penalties.

#### 2. Critical Directives
*   **`/arch:AVX512`**: Forces the compiler to emit 512-bit wide instructions (ZMM registers). This is the engine's primary mechanism for achieving "zero-cost" compute, enabling 16x FP32 operations per cycle per core.
*   **`/GS-` (Buffer Security Check Disable)**: A critical optimization for high-performance kernels. By disabling stack-based buffer overflow checks, it removes the function prologue/epilogue overhead, essential for tight, heapless loops where every cycle counts.
*   **`/Oi /Ot /Ox`**: Enables intrinsic function expansion and aggressive optimization for speed. This forces the compiler to inline math primitives directly into the instruction stream, preventing the overhead of function calls within the neural network's hot paths.
*   **`find_package(Torch)`**: Integrates LibTorch strictly as a build-time dependency for tensor layout definitions and weight serialization, ensuring the engine remains decoupled from the heavy PyTorch runtime during inference.

#### 3. Contribution to Heapless/Zero-Cost Architecture
*   **Memory Determinism**: By isolating the `nn` sub-module, the architecture enforces a strict separation between the model definition and the execution engine. This allows for the implementation of **Arena Allocation** or **Stack-based Tensor Buffers**, preventing heap fragmentation and non-deterministic latency spikes.
*   **Instruction-Level Parallelism (ILP)**: The combination of C++20 standard compliance and AVX-512 flags allows the compiler to perform auto-vectorization on the engine's core kernels. This transforms high-level neural operations into contiguous, cache-friendly memory access patterns.
*   **Zero-Overhead Abstractions**: The build configuration supports the use of `constexpr` and template metaprogramming within the `nn` sub-module. By resolving neural network topologies at compile-time, the engine eliminates the need for runtime graph traversal, effectively reducing the "Neural Engine" to a series of pre-computed, hardware-optimized instruction blocks.

---

## File: `autonomous_optimizer.py`

### Architectural Analysis: CENTAUR Autonomous Optimizer

The `autonomous_optimizer.py` acts as a **closed-loop, LLM-driven JIT-refactoring agent** designed to enforce high-performance constraints (AVX-512, heapless memory management) on the CENTAUR Neural Engine. It treats the C++ codebase as a mutable state machine, using compiler feedback as the objective function for optimization.

#### Core Purpose
To automate the transition from idiomatic C++ to hardware-aligned, SIMD-vectorized code. It enforces "Zero-Cost" abstractions by iteratively applying intrinsic-heavy transformations and validating them against the project's CTest suite, ensuring that performance gains do not compromise mathematical integrity.

#### Critical Function Breakdown

*   **`propose_target`**: Performs semantic analysis of the `PROJECT_DOCUMENTATION_AND_FUNCTIONS.md` (Grapify map). It identifies high-entropy code paths—specifically those involving heavy matrix operations or scalar loops—that are prime candidates for AVX-512 vectorization.
*   **`safe_optimize`**: Executes the primary transformation logic. It forces the LLM to replace scalar logic with `__m512` / `__m512i` intrinsics, enforcing strict alignment requirements and removing heap-allocated containers (e.g., `std::vector`) in favor of stack-allocated or pre-allocated buffer patterns.
*   **`attempt_fix`**: A specialized self-healing routine. It performs differential analysis between the original code and the failed optimization, specifically targeting common SIMD pitfalls: type-mismatches in intrinsic registers, unaligned memory access (e.g., `_mm512_loadu_ps` vs `_mm512_load_ps`), and precision loss in floating-point reductions.
*   **`apply_and_test`**: The validation gate. It bridges the gap between the LLM's generative output and the hardware reality. By invoking `cmake` and `ctest` in a controlled environment, it ensures that the generated code satisfies the strict binary requirements of the CENTAUR engine.

#### Contribution to CENTAUR Physics Architecture

1.  **SIMD-First Enforcement**: By mandating AVX-512, the agent forces the codebase to maintain 512-bit alignment, which is critical for the throughput requirements of the Neural Engine.
2.  **Heapless Memory Discipline**: The agent acts as a static analyzer that discourages dynamic allocation, pushing the architecture toward deterministic, stack-based memory layouts that minimize cache misses and latency jitter.
3.  **Automated Regression Testing**: The "Self-Healing" loop creates a robust development cycle where performance optimizations are verified against the existing test suite, preventing the "optimization-induced bug" phenomenon common in manual SIMD refactoring.
4.  **Continuous Refinement**: The agent treats the codebase as a living entity, allowing for iterative improvements as the underlying hardware-specific documentation (Grapify) evolves.

---

## File: `generate_doc.py`

### Architectural Analysis: `generate_doc.py`

**Core Purpose**
This script serves as an automated documentation synthesis engine for the CENTAUR codebase. It acts as a metadata extraction layer, leveraging LLM-based semantic analysis to map high-level architectural intent to low-level C++ implementation details. By automating the generation of technical documentation, it ensures that the "zero-cost, heapless" design philosophy remains documented without manual overhead.

**Critical Functions**

*   **`process_file(filepath, chain)`**: Implements a robust, state-aware ingestion pipeline. It features an exponential backoff retry mechanism specifically tuned for high-latency API interactions, ensuring that the documentation process is resilient against rate-limiting (429) during large-scale codebase scans.
*   **`main()` (Orchestration)**: Manages the file-system traversal and concurrency model. It enforces strict exclusion policies (e.g., `build`, `tests`, `deployment`) to ensure the documentation focuses exclusively on the core engine logic rather than transient build artifacts or validation suites.
*   **`ThreadPoolExecutor` Integration**: Provides a controlled concurrency model (limited to 3 workers) to balance throughput against API quota constraints, maintaining the integrity of the documentation stream while minimizing total execution time.

**Contribution to CENTAUR Architecture**

*   **Zero-Cost Documentation**: By offloading the analysis to an LLM, the engine maintains high-fidelity documentation without requiring developers to write verbose comments that could potentially bloat binary sizes or introduce maintenance debt.
*   **Heapless Integrity**: The script operates entirely outside the runtime environment. By keeping the documentation generation decoupled from the C++ source, it ensures that the CENTAUR engine remains strictly heapless and free of runtime overhead, as no documentation-related logic is compiled into the final binary.
*   **Architectural Mapping**: The prompt template forces the LLM to focus on the "AVX-512 physics architecture," ensuring that the generated output highlights SIMD-specific optimizations, register usage, and memory alignment strategies. This creates a searchable, human-readable index of the engine's performance-critical paths, facilitating easier auditing of the zero-cost abstractions.
*   **Deduplication Strategy**: The explicit instruction to avoid boilerplate ensures that the resulting `PROJECT_DOCUMENTATION_AND_FUNCTIONS.md` remains a high-signal document, focusing on unique algorithmic contributions rather than generic code descriptions.

---

## File: `multi_agent_conversation.py`

### Architectural Analysis: NCA Graph-Curve vs. KV-Cache

This script serves as a validation harness for the **CENTAUR Neural Engine’s** core hypothesis: that transformer inference can be decoupled from the $O(N)$ memory growth of the KV-cache by mapping weight-space activations onto a fixed-topology graph-curve.

#### 1. Core Architectural Purpose
The script demonstrates **Weight-Preserving Inference (WPI)**. By loading the same `Gemma-4` GGUF weights into two distinct execution contexts, it isolates the serving architecture as the sole variable. The "NCA Graph-Curve" engine replaces the standard KV-cache (which grows linearly with sequence length) with a recurrent state mechanism that maintains a constant memory footprint, effectively treating the transformer as a state-space model (SSM) without requiring retraining.

#### 2. Critical Functional Components
*   **`llm_original` (Standard KV-Cache):** Acts as the baseline. It utilizes a dynamic heap-allocated buffer for attention keys and values. As `NUM_TURNS` increases, the memory pressure scales $O(N)$, leading to cache-miss degradation and eventual latency spikes as the context window fills.
*   **`llm_nca` (Graph-Curve Engine):** Implements a fixed-window recurrent state. By constraining `n_ctx=512`, it forces the engine to compress historical context into a static state vector. This is the software-level abstraction of our **heapless AVX-512 pipeline**, where state updates are performed via register-resident accumulation rather than memory-bound pointer chasing.
*   **Prompt Injection Loop:** The cross-talk mechanism (`last_nca_response` fed to `orig_prompt`) tests the semantic stability of the NCA engine. It verifies that the graph-curve state maintains coherence equivalent to the full KV-cache, proving that the "physics" of the weights are preserved despite the architectural shift.

#### 3. Contribution to CENTAUR AVX-512 Physics
The NCA architecture is designed to map directly to **AVX-512 FMA (Fused Multiply-Add) units**:

*   **Deterministic Latency:** By eliminating the KV-cache, we remove the non-deterministic memory access patterns (pointer chasing in the attention matrix). This allows the CENTAUR engine to schedule instructions with cycle-accurate precision.
*   **Register-Resident State:** The "Graph-Curve" approach allows the recurrent state to reside entirely within the ZMM register file. This avoids the L1/L2 cache hierarchy bottlenecks inherent in standard transformer serving.
*   **Zero-Cost Transition:** Because the weights are identical, the NCA engine functions as a **hardware-accelerated projection layer**. It treats the GGUF weights as static kernels, applying them to the recurrent state using `_mm512_fmadd_ps` instructions, effectively turning the transformer into a high-throughput signal processing pipeline.

**Architectural Verdict:** The script validates that the NCA engine achieves $O(1)$ memory complexity while maintaining the semantic integrity of the original model. This is the prerequisite for our goal: a pure C++ implementation where the entire inference loop is unrolled into a single, branchless AVX-512 execution block.

---

## File: `nn\CMakeLists.txt`

### Architectural Analysis: NCA Root CMake

The `CMakeLists.txt` serves as the foundational enforcement layer for the **Nano-Core Architecture (NCA)**. It transitions the build environment from standard LibTorch-heavy paradigms to a strict, high-performance AVX-512 execution model.

#### 1. Standards Enforcement & ABI Integrity
*   **C++20 Strictness:** By forcing `/Zc:__cplusplus` and `/permissive-`, the build ensures that template metaprogramming and concepts—critical for zero-cost abstraction in SIMD dispatch—are evaluated against the C++20 standard rather than legacy MSVC behaviors.
*   **LibTorch Sanitization:** The aggressive stripping of `/std:c++17` flags is vital. LibTorch often injects legacy flags that conflict with C++20 modules or `std::span` usage, which are essential for the heapless memory management required by the CENTAUR engine.

#### 2. Zero-Cost SIMD Dispatch Strategy
*   **Header-Only Topology:** The `include_directories` configuration establishes a flat, root-relative namespace. This facilitates the "Zero-Cost" design by allowing the compiler to inline SIMD kernels across translation units without the overhead of dynamic linking or opaque binary interfaces.
*   **Warning Suppression:** The suppression of `C4244` and `C4267` is a calculated architectural trade-off. In high-performance AVX-512 kernels, explicit narrowing casts are often required for register-width alignment; suppressing these warnings prevents "noise" while relying on static analysis to ensure register safety.

#### 3. Heapless Memory Alignment
*   **Compiler-Level Constraints:** By enforcing `/EHsc` and strict conformance, the build environment prevents the injection of non-deterministic runtime exceptions or hidden heap allocations often triggered by implicit standard library conversions.
*   **AVX-512 Readiness:** The configuration is primed for `core/simd/dispatch.hpp`. By centralizing the include path, the architecture ensures that the compiler can resolve SIMD intrinsics at compile-time, enabling the "Nano-Core" to treat memory buffers as raw, aligned pointers rather than managed objects.

#### 4. Critical Architectural Directives
| Directive | Purpose |
| :--- | :--- |
| `/Zc:preprocessor` | Ensures token-pasting and macro expansion are standards-compliant, preventing subtle bugs in SIMD intrinsic wrappers. |
| `/utf-8` | Guarantees consistent character encoding for cross-platform kernel metadata, preventing build-time non-determinism. |
| `CMAKE_CXX_EXTENSIONS OFF` | Disables compiler-specific extensions (e.g., GNU extensions in MSVC), ensuring the codebase remains portable and strictly compliant with the AVX-512 intrinsic set. |

**Summary:** This file acts as a "hardened shell." It strips away the bloat of the LibTorch ecosystem to expose a lean, C++20-compliant environment where the compiler can aggressively optimize SIMD kernels without the interference of legacy runtime behaviors.

---

## File: `nn\config\model_config.hpp`

### Architectural Analysis: `model_config.hpp`

This header defines the static memory layout and execution parameters for the **CENTAUR Neural Engine**. By pinning these values as `constexpr`, the engine eliminates runtime configuration overhead, enabling the compiler to bake tensor shapes and routing logic directly into the AVX-512 instruction stream.

#### 1. Memory-Mapped Tensor Alignment
*   **`D_MODEL = 2048`**: Defines the primary vector width for the engine. At 2048 dimensions, this aligns perfectly with 512-bit AVX-512 registers (processing 16 `float32` elements per instruction). This ensures cache-line-friendly access patterns and minimizes register spilling during GEMM operations.
*   **`VISION_FEATURE_GRID` & `CHANNELS`**: These constants dictate the static allocation of the vision buffer. By fixing these, the engine avoids dynamic heap allocation, allowing for stack-based or static-data-segment buffer management, critical for deterministic latency.

#### 2. SDMS (Saliency-Driven Mixture of Experts)
*   **`N_MICRO_EXPERTS = 1024`**: Implements a fine-grained MoE architecture. The high count allows for sparse activation, where only `TOP_K_EXPERTS` (16) are loaded into the L1/L2 cache.
*   **`SALIENCY_THRESHOLD`**: Acts as a hardware-level gate. By using a static threshold, the router can implement a branchless `vcmpps` (compare packed single-precision) instruction to mask out inactive experts, preventing unnecessary compute cycles on low-entropy tokens.

#### 3. Spectral RLS (Recursive Least Squares)
*   **`KRONECKER_FACTOR_DIM`**: Defines the rank-decomposition of the memory state. By using Kronecker factors (64x32), the engine performs matrix updates via smaller, cache-resident sub-matrices rather than full-rank updates, reducing the complexity of the RLS memory state from $O(d^2)$ to $O(a+b)$.
*   **`RLS_FORGETTING_FACTOR`**: Controls the temporal decay of the hidden state. This is implemented as a constant multiplier in the AVX-512 fused-multiply-add (FMA) pipeline, ensuring the "long memory" update is a single-cycle operation.

#### 4. CENTAUR Execution Engine
*   **`LogicBackend::CENTAUR_Compiled`**: This is the core of the zero-cost abstraction. When selected, the engine bypasses generic dispatch tables. Instead, it uses template metaprogramming to generate specialized kernels that inline the `ACT_HALT_THRESHOLD` and `MAX_ACT_CYCLES` logic directly into the inference loop.
*   **Heapless Strategy**: By defining all dimensions and thresholds as `constexpr`, the engine allows the compiler to perform **Static Buffer Planning**. The entire model state is effectively a fixed-size memory map, allowing the AVX-512 kernels to operate on pre-allocated, aligned memory blocks without ever invoking `malloc` or `new`.

#### 5. Critical Performance Implications
*   **Branchless Routing**: The `TOKEN_` registry and `LogicBackend` enum allow the compiler to resolve routing paths at compile-time.
*   **Instruction Pipelining**: The `MAX_ACT_CYCLES` and `DEEP_ACT_CYCLES` constants define the loop unrolling factor for the recursive activation logic. This allows the AVX-512 backend to saturate the execution ports by unrolling the "thinking" cycles, minimizing loop-control overhead.

---

## File: `nn\core\CMakeLists.txt`

### Architectural Analysis: `nca_core`

The `nca_core` library serves as the hardware-abstraction layer for the CENTAUR Neural Engine, bridging high-level tensor operations with bare-metal AVX-512 execution. It enforces a **heapless, deterministic execution model** by offloading compute to pre-allocated static memory buffers managed by the `silicon_memory` subsystem.

#### 1. Core Subsystems
*   **Spectral Logic (`spectral/`):** Implements Fast Walsh-Hadamard Transforms (FWHT) and Kronecker RLS. These replace traditional dense matrix multiplications with O(N log N) complexity, leveraging AVX-512 permute/shuffle instructions to minimize cache-line stalls.
*   **Execution/Wavefront Routing (`execution/`):** Manages the instruction pipeline. The `wavefront_router` orchestrates data-flow across the AVX-512 register file, ensuring that the `importance` module can prune compute-heavy branches before they hit the execution units.
*   **SIMD Dispatch (`simd/`):** A multi-target dispatch layer. It uses runtime CPUID checks to fallback from AVX-512 (VNNI/BW) to AVX2, ensuring zero-cost degradation on legacy silicon.

#### 2. Critical Architectural Components
*   **`silicon_memory.cpp`:** Implements a static memory arena. By bypassing the standard heap, it eliminates non-deterministic latency spikes caused by page faults or allocator contention, critical for real-time neural inference.
*   **`avx512_kernels.cpp`:** The primary compute engine. It utilizes `_mm512_mask_add_ps` and `_mm512_dpbusd_epi32` (VNNI) to perform fused-multiply-accumulate operations directly on 512-bit wide registers, maximizing throughput per clock cycle.
*   **`halting.cpp`:** Implements adaptive compute depth. It monitors the convergence of the neural state and triggers early-exit logic, effectively reducing the total FLOP count per inference pass.

#### 3. Zero-Cost Physics Integration
The architecture achieves "zero-cost" by enforcing **data-locality constraints**:
*   **Compile-time Specialization:** The `target_compile_options` force the compiler to emit AVX-512 instructions globally, allowing the optimizer to inline kernels directly into the `route_planner` logic.
*   **Register-Level Pipelining:** By avoiding heap allocations, the `nca_core` keeps the working set within the L1/L2 cache hierarchy. The `spectral_pruner` ensures that only high-entropy features are processed, preventing the AVX-512 units from wasting cycles on zero-valued activations.
*   **Deterministic Scheduling:** The `silicon_automation` layer treats the neural network as a static graph, mapping layers to specific AVX-512 register banks to eliminate register spilling and stack-based memory access.

---

## File: `nn\core\activations.cpp`

### Architectural Analysis: `nn/core/activations.cpp`

This module serves as the high-performance activation layer for the CENTAUR Neural Engine, implementing the SiLU (Swish) activation function. It prioritizes instruction-level parallelism (ILP) and hardware-specific dispatch to minimize latency in the compute pipeline.

#### 1. Core Purpose
The module provides a unified interface for non-linear activation, abstracting hardware-specific SIMD implementations from the high-level neural graph. By leveraging `std::span`, it enforces memory safety without heap allocation, maintaining the engine's "zero-cost" abstraction mandate.

#### 2. Critical Functions
*   **`silu_scalar`**: Acts as the fallback execution path. The 4x loop unrolling minimizes branch misprediction penalties and maximizes pipeline utilization for non-AVX-capable hardware. It serves as the baseline for verifying SIMD kernel correctness.
*   **`silu`**: The primary entry point. It utilizes the `NCA_DISPATCH_KERNEL` macro to perform runtime CPUID feature detection. This ensures the engine executes the most efficient instruction set (AVX-512 > AVX2 > Scalar) without requiring manual developer intervention.

#### 3. Architectural Integration
*   **Zero-Cost Abstraction**: The use of `std::span` avoids `std::vector` overhead, ensuring the function operates directly on pre-allocated buffers (typically managed by the engine's static memory arena). This eliminates heap fragmentation and pointer chasing.
*   **AVX-512 Synergy**: By offloading to `simd::avx512::silu`, the architecture exploits 512-bit wide registers, allowing for 16 single-precision float operations per cycle. This is critical for the CENTAUR engine's throughput requirements, as it maximizes the ratio of compute-to-memory-bandwidth.
*   **Branch Prediction Optimization**: The `[[likely]]` attribute on the unrolled scalar loop provides a hint to the compiler to prioritize the hot path, reducing instruction cache pressure and ensuring the pipeline remains saturated during scalar fallback.
*   **Memory Alignment**: The architecture assumes `data` is aligned to 64-byte boundaries (AVX-512 requirement). The dispatch mechanism implicitly relies on the caller to provide aligned buffers, maintaining the "pure C++" performance profile by avoiding costly unaligned load/store penalties.

---

## File: `nn\core\activations.hpp`

### Architectural Analysis: `nn/core/activations.hpp`

#### Core Purpose
This header defines the interface for the **NCA (Neural Compute Architecture) Activation Layer**, serving as the primary abstraction for non-linear transformation kernels. It acts as a dispatch gateway, decoupling high-level neural graph execution from hardware-specific SIMD implementations.

#### Critical Function: `silu(std::span<float>)`
*   **Mechanism:** Implements the Sigmoid-Weighted Linear Unit ($x \cdot \sigma(x)$).
*   **Memory Model:** Operates strictly in-place on `std::span<float>`, enforcing zero-copy semantics. By utilizing `std::span`, the function maintains bounds safety without heap allocation or pointer decay, ensuring compatibility with stack-allocated or pre-mapped memory buffers.
*   **SIMD Dispatch:** The implementation is designed to resolve at compile-time (via template specialization or `if constexpr` dispatch) to AVX-512 intrinsic sequences.

#### Contribution to Zero-Cost/Heapless Architecture
*   **Zero-Cost Abstraction:** The use of `std::span` provides a lightweight view over contiguous memory, eliminating the overhead of `std::vector` or custom container metadata. It allows the compiler to perform aggressive loop unrolling and vectorization (VMOVUPS/VMOVAPS) without pointer aliasing concerns.
*   **Heapless Execution:** By design, the function signature precludes dynamic allocation. It assumes the caller provides a pre-allocated buffer, aligning with the "Arena" or "Scratchpad" memory model required for high-performance inference.
*   **AVX-512 Integration:** The architecture leverages 512-bit registers to process 16 `float` elements per instruction. The SiLU implementation utilizes `_mm512_exp_ps` (or a high-precision polynomial approximation) and `_mm512_div_ps` (or reciprocal estimates) to minimize latency, ensuring the activation layer does not become a bottleneck in the compute pipeline.
*   **In-Place Transformation:** By modifying the buffer in-place, the architecture maximizes L1/L2 cache locality, reducing memory bus pressure—a critical requirement for maintaining throughput in the CENTAUR Neural Engine.

---

## File: `nn\core\centaur\centaur.hpp`

### Architectural Analysis: CENTAUR Neural Engine

CENTAUR is a **hardware-specialized JIT-compilation engine** designed to eliminate runtime branch misprediction and memory latency overhead in high-throughput inference. It treats the CPU as a fixed-function data-flow processor rather than a general-purpose execution unit.

#### 1. Core Architectural Philosophy
*   **Static Determinism:** By moving all decision-making (tiling, prefetching, scheduling) to the `GraphBuilder` phase, the `step()` function becomes a branchless, linear traversal of a pre-compiled DAG.
*   **Hardware-Adaptive Mapping:** The engine fingerprints the microarchitecture (L1/L2/L3 cache lines, SIMD width, AMX availability) to generate an `ExecutionPlan` that aligns memory access patterns with the specific cache hierarchy of the host.
*   **Zero-Dynamic Overhead:** The "heapless" nature is achieved by pre-allocating a fixed `arena_size_bytes` during the compilation phase, ensuring that the hot path (`step`) performs no dynamic memory allocation or pointer chasing.

#### 2. Critical Component Breakdown
*   **`CPUProfile` (Fingerprinting):** Captures the physical constraints of the silicon. This is the source of truth for the `TilePlanner`, ensuring that GEMM kernels are blocked to fit perfectly within the L1d/L2 cache to minimize cache-miss stalls.
*   **`GraphBuilder` (Compilation):** Transforms high-level model parameters into a `ExecutionGraph`. It performs the heavy lifting of calculating optimal `mr` (micro-kernel register) and `nr` (register) blocking factors for AVX-512/AMX.
*   **`GraphExecutor` (Execution):** The runtime engine. It executes the `ExecutionGraph` as a flat loop. By using `PREFETCH` nodes, it hides memory latency by interleaving data movement with compute, effectively saturating the AVX-512 execution units.
*   **`ExecutionPlan`:** Encapsulates the hardware-specific strategy. It defines the `μKernel` dimensions, which are critical for maximizing the throughput of the AVX-512 FMA (Fused Multiply-Add) pipelines.

#### 3. AVX-512/Systems Integration
*   **SIMD Alignment:** The engine forces data alignment to 64-byte boundaries, matching the AVX-512 cache line size, preventing split-load penalties.
*   **Prefetch Strategy:** The `ExecutionGraph` explicitly encodes `prefetch_target_slot` and `prefetch_hint`. This allows the engine to issue `_mm_prefetch` instructions ahead of the compute nodes, ensuring the L1 cache is "warm" before the `GEMM_TILE` kernel begins.
*   **Spectral/BCT Kernels:** The inclusion of `SPECTRAL_FWHT` and `UNIFIED_BCT` (Block-Compressed Tiling) suggests an architecture optimized for non-standard, high-efficiency neural operations that bypass traditional dense matrix multiplication bottlenecks.

#### 4. Performance Implications
*   **Latency:** By eliminating dynamic dispatch and runtime graph interpretation, the engine achieves near-theoretical peak FLOPs for the given hardware.
*   **Throughput:** The `load_weights` and `load_state` methods imply a decoupled memory-to-register pipeline, allowing the engine to maintain high occupancy of the AVX-512 register file across the entire `step()` execution.

**Summary:** CENTAUR is a **compile-time-first** architecture. It treats the model as a static hardware configuration, effectively "baking" the neural network into the CPU's execution pipeline.

---

## File: `nn\core\centaur\cpu_fingerprint.hpp`

### Architectural Analysis: `cpu_fingerprint.hpp`

The `cpu_fingerprint.hpp` module serves as the **Hardware-Abstraction Layer (HAL) root** for the CENTAUR engine. It implements a "Hardware-First" execution model, where the binary's runtime behavior is strictly constrained by the physical silicon's topology rather than abstract software heuristics.

#### 1. Core Purpose
*   **Deterministic Hardware Mapping:** By probing cache hierarchies and SIMD capabilities at startup, the engine enables **JIT-like specialization** without the overhead of a JIT compiler.
*   **Control Inversion:** It enforces a contract where the execution topology (tiling, blocking, and vectorization strategy) is derived from the `CPUProfile` singleton, ensuring that memory access patterns are always cache-aligned and SIMD-width aware.

#### 2. Critical Mechanisms
*   **CPUID-Driven Topology Discovery:** Uses `leaf 4` (Intel) and `0x8000001D` (AMD) to reconstruct the cache hierarchy. This allows the engine to calculate optimal **GEMM tile sizes** that fit within L1/L2 boundaries, minimizing cache thrashing.
*   **XSAVE-Aware SIMD Validation:** Beyond simple feature flags, it validates OS-level state management (`XGETBV`). This is critical for AVX-512/AMX, as it prevents illegal instruction faults if the OS kernel has not enabled the ZMM/Tile register states.
*   **Heuristic Bandwidth Modeling:** Provides a per-core memory bandwidth estimate. This is used by the scheduler to decide between **compute-bound** (AVX-512 heavy) and **memory-bound** (prefetch-heavy) execution paths.

#### 3. Contribution to Zero-Cost/Heapless Architecture
*   **Static Singleton Pattern:** The `fingerprint()` function uses a thread-safe static initialization, ensuring the profile is computed exactly once. This eliminates runtime branching and heap allocations during the hot-path execution.
*   **Compile-Time Constants:** By exposing `constexpr` methods (e.g., `simd_width_f32()`, `register_file_floats()`), the engine allows the compiler to perform **constant folding** on loop unrolling factors and register pressure calculations.
*   **Zero-Allocation Topology:** The profile is a POD (Plain Old Data) structure. It provides the necessary metadata for the engine to perform **stack-based buffer management**, ensuring that all neural network activations and weights are mapped to memory regions that respect the physical cache line size (64 bytes) and associativity discovered at startup.

#### 4. Architectural Risks & Mitigation
*   **Fallback Logic:** The inclusion of "Ultimate fallback" values ensures the engine remains functional on legacy or virtualized hardware where CPUID might be masked or incomplete.
*   **AMX/AVX-512 Integration:** The profile explicitly separates `amx_tile` from `avx512f`. This allows the CENTAUR engine to switch between **Vector-based (AVX-512)** and **Matrix-based (AMX)** kernels based on the specific silicon generation (e.g., Sapphire Rapids vs. Ice Lake), maximizing throughput per watt.

---

## File: `nn\core\centaur\execution_graph.hpp`

### Architectural Analysis: CENTAUR Execution Graph

The `execution_graph.hpp` module serves as the **static orchestration layer** for the CENTAUR Neural Engine. It transforms high-level neural operations into a cache-aware, branchless execution sequence, effectively turning the CPU into a deterministic ASIC-like pipeline.

#### 1. Core Architectural Philosophy
*   **Heapless Execution:** By pre-allocating the `ExecutionGraph` and its associated memory arena at startup, the engine eliminates runtime `malloc`/`free` jitter.
*   **SoA (Structure-of-Arrays) Layout:** The `ExecutionGraph` uses parallel arrays for node metadata. This ensures that when the engine traverses the graph, it performs **unit-stride, cache-line-aligned loads**, maximizing L1 cache line utilization and enabling the hardware prefetcher to predict the next node's metadata with 100% accuracy.
*   **Zero-Branch Dispatch:** By resolving `KernelFn` pointers at compile-time, the engine replaces dynamic dispatch (virtual functions/if-else chains) with a direct function pointer jump, minimizing branch misprediction penalties in the hot path.

#### 2. Critical Components
*   **`ExecutionGraph` (The Static DAG):** A packed, 64-byte aligned structure. The use of `uint16_t` for slot indexing and `uint32_t` for payloads ensures the entire graph fits within a minimal memory footprint, keeping the "instruction stream" of the neural network resident in L1/L2.
*   **`GraphBuilder` (The Neural Compiler):** A pure function factory that maps `CPUProfile` to a hardware-specific execution sequence. It performs the "heavy lifting" of memory slot allocation and kernel resolution, ensuring the runtime only performs pointer dereferencing.
*   **`emit_` Methods:** These act as the "assembler" for the neural engine, serializing high-level operations into the `ExecutionGraph`'s SoA buffers.

#### 3. AVX-512 Optimization Strategy
*   **Cache Pinning:** The `slot_pinned` and `slot_cache_level` metadata allow the engine to explicitly manage data residency. By pinning weight slabs (e.g., GLR alpha/beta) to L2, the engine ensures that compute-heavy kernels (GEMM/BCT) operate on data that is already "warm."
*   **Payload Packing:** The `payload_0/1/2` fields are designed for direct consumption by AVX-512 registers. The engine can load a full node's metadata into a single `ZMM` register, allowing the dispatcher to process node parameters using SIMD-parallel logic if necessary.
*   **Unified BCT Engine Integration:** By embedding the `UnifiedBCTEngine` within the graph, the compiler treats routing and expert execution as a single, contiguous memory-access pattern, preventing the "stutter" typically found in MoE implementations when switching between gating and expert weights.

#### 4. Performance Impact
*   **Deterministic Latency:** Because the graph is static and sequential, the execution time per inference pass is constant, enabling strict real-time guarantees.
*   **Instruction Cache Efficiency:** The linear traversal of the `ExecutionGraph` arrays ensures that the CPU's instruction fetcher encounters no jumps or complex control flow, keeping the execution pipeline saturated with compute-heavy kernel instructions rather than control-flow overhead.

---

## File: `nn\core\centaur\graph_executor.hpp`

### Architectural Analysis: `GraphExecutor`

The `GraphExecutor` serves as the deterministic runtime engine for the CENTAUR neural architecture. It operates as a **non-branching instruction sequencer** that maps a pre-compiled `ExecutionGraph` onto a cache-aligned memory arena, effectively treating the CPU as a fixed-function hardware accelerator.

#### Core Architectural Principles
*   **Zero-Dynamic Dispatch:** The `execute()` loop is a flat, linear traversal of `ExecNode` structures. By resolving all graph dependencies at compile-time, it eliminates runtime graph traversal overhead and branch misprediction penalties.
*   **Arena-Centric Memory Model:** All tensors reside in a single, 64-byte aligned `arena_`. This ensures cache-line alignment for AVX-512 vector loads/stores and minimizes TLB pressure by keeping the working set contiguous.
*   **Hardware-Pinned Execution:** The executor leverages explicit `_mm_prefetch` hints and `_mm_sfence` barriers to manage the memory hierarchy, ensuring that data is staged in L1/L2 caches before the compute kernels (GLR, BCT) trigger.

#### Critical Function Breakdown

*   **`execute()` (The Sequencer):** Acts as the primary dispatch loop. The use of `[[likely]]` attributes and the absence of complex logic inside the loop body allows the compiler to unroll the execution path, effectively turning the graph into a static sequence of SIMD instructions.
*   **`execute_glr()` (Gated Linear Recurrence):** The performance-critical path. It utilizes a **blocked AVX-512 FMA loop** (processing 64 floats per iteration) to maximize throughput. It incorporates manual software prefetching for the next iteration's weights, hiding memory latency behind the FMA pipeline.
*   **`execute_rmsnorm()`:** Implements a two-pass normalization. The first pass uses `_mm512_reduce_add_ps` to compute the sum of squares, minimizing horizontal dependency chains. The second pass applies the reciprocal square root scaling via vectorized multiplication.
*   **`execute_unified_bct()`:** Provides an abstraction layer for the BCT (Batched Expert) engine. By delegating to a pre-initialized engine, it maintains the executor's "flat" structure while allowing for complex, expert-specific GEMM kernels that are likely L2-pinned.
*   **`execute_prefetch()`:** A proactive cache management utility. It translates high-level graph hints into specific `_MM_HINT_T0/T1/NTA` instructions, ensuring that the data-hungry GLR and BCT kernels never stall on DRAM fetches.

#### Contribution to "Heapless" Physics Architecture
The executor avoids dynamic memory allocation during the inference loop. By pre-allocating the `arena_` in the constructor and using `slot_ptr` for pointer arithmetic, it ensures that the execution phase is **strictly deterministic**. This eliminates non-deterministic latency spikes caused by heap fragmentation or garbage collection, satisfying the requirements for high-frequency, low-jitter neural inference.

---

## File: `nn\core\centaur\tile_planner.hpp`

### Architectural Analysis: `tile_planner.hpp`

The `tile_planner` acts as the **hardware-abstraction layer** for the CENTAUR engine. It transforms static model dimensions into a dynamic, silicon-aware execution schedule, effectively treating the CPU cache hierarchy as a programmable memory controller.

#### 1. Core Purpose: Hardware-Model Decoupling
The module implements a **"Silicon-First"** paradigm. Instead of relying on model-defined hyperparameters, it performs a static analysis of the `CPUProfile` (L1/L2/L3 capacities, cache-line sizes, and SIMD width) to derive optimal tiling and prefetching strategies. This ensures the compute kernels are always operating at the peak of the machine's roofline model.

#### 2. Critical Logic Components
*   **`plan_execution` (The Deterministic Planner):** A pure, heapless function that maps the physical constraints of the processor to an `ExecutionPlan`. By calculating `M` (rows) based on `L1d` occupancy and `K` (reduction) based on `d_model`, it ensures the working set never exceeds the L1 cache, minimizing expensive DRAM round-trips.
*   **Strategy Classification:** Dynamically selects the execution mode (`L1_RESIDENT` to `DRAM_BANDWIDTH`). This allows the engine to switch from compute-bound register-tiling to memory-bound streaming prefetch patterns without modifying the underlying kernel code.
*   **MoE Cache Compilation:** The `ExpertBlockLayout` logic treats MoE experts as cache-resident slabs. By calculating `num_resident_experts` based on `L2` capacity, it enables "cache-aware routing," where the engine prioritizes loading experts that are already resident in the L2, significantly reducing latency in sparse MoE inference.

#### 3. Contribution to Zero-Cost/Heapless Architecture
*   **Zero-Cost Abstraction:** The planner is designed to be called once at kernel initialization. The resulting `ExecutionPlan` is a POD (Plain Old Data) struct, allowing the compiler to inline the plan parameters directly into the assembly of the microkernels.
*   **Heapless Execution:** By using stack-allocated structs and deterministic sizing, the engine avoids runtime memory allocation during the inference hot path.
*   **AVX-512 Optimization:** The planner explicitly accounts for the 32-register file of AVX-512, setting the `mr` (micro-rows) and `nr` (micro-cols) to maximize register pressure and FMA throughput, while using `kr` (unroll depth) to hide pipeline latency.

#### 4. Architectural Insight
The most significant innovation is the **prefetch depth scaling**. By tying `prefetch.depth` and `use_nt_stores` to the `L3` size and bandwidth, the planner effectively implements a software-defined memory controller. It shifts from standard cache-line loads to Non-Temporal (NT) stores when the working set exceeds the L3, preventing cache pollution and bypassing the L1/L2 hierarchy for write-heavy operations.

---

## File: `nn\core\centaur\unified_bct_engine.hpp`

### Architectural Analysis: `UnifiedBCTEngine`

The `UnifiedBCTEngine` implements a **Binary Curve Tree (BCT)** architecture, replacing traditional dense matrix multiplication (GEMM) with a bit-manipulation-based phase-collapse routing mechanism. It treats neural weights as binary sign-flip operators, effectively transforming the inference path into a series of XOR-based accumulation cycles.

#### 1. Core Architectural Philosophy
*   **Phase-Collapse Routing:** Instead of a Softmax-gated Top-K router, the engine uses a single AVX-512 `_mm512_cmp_ps_mask` operation to project the input vector into a $B$-dimensional binary space. This collapses the routing decision into a bitmask, eliminating the need for expensive activation functions and sorting networks.
*   **Zero-Cost Memory Layout:** The engine utilizes a single, cache-aligned `arena_` allocation. By mapping `router_w_down_` and `experts_arena_` into a contiguous block, it ensures spatial locality and minimizes TLB misses during the high-frequency execution loop.
*   **XOR-Accumulation (Sign-Flip Logic):** The engine replaces floating-point multiplication with `_mm512_xor_si512` operations. By treating weights as sign-flip masks, it performs "multiplication" by toggling the sign bit (bit 31) of the input floats, effectively performing a hardware-native sign-magnitude transformation.

#### 2. Critical Function Breakdown
*   **`initialize_random`:** Implements a deterministic Xorshift-based weight initialization. It enforces a scale factor derived from the BCT depth ($B$), ensuring the variance of the phase-collapse projection remains stable across the binary manifold.
*   **`execute` (Phase 1: Routing):** Performs a vectorized dot-product against the router weights. The use of `_mm512_fmadd_ps` followed by a mask-based selection (`k1`) allows the engine to identify the primary expert path in $O(B)$ time. The secondary expert (`k2`) is derived via a bit-flip of the minimum-magnitude component, maintaining a continuous-discrete hybrid state.
*   **`execute` (Phase 2: Expert Execution):** Executes the discrete expert logic using the XOR-sign-flip pattern.
    *   **Input Projection:** Uses `_mm512_slli_epi32` to shift the weight bitmask into the sign-bit position, applying the transformation via `_mm512_xor_si512`.
    *   **Output Accumulation:** Re-applies the sign-flip logic to the expert outputs before adding them to the global output buffer, maintaining the "zero-multiplier" constraint throughout the entire inference pipeline.

#### 3. Systems Impact
*   **Arithmetic Efficiency:** By eliminating FP32 multipliers, the engine shifts the bottleneck from the FPU's multiplier throughput to the load/store and XOR latency of the AVX-512 unit. This significantly reduces power consumption and thermal throttling on high-density inference nodes.
*   **Heapless Operation:** The engine is designed for embedded or high-throughput server environments where dynamic allocation is prohibited post-initialization. The `arena_` approach ensures that the entire model state is a single, contiguous memory object, facilitating easy serialization and zero-copy loading.
*   **Hardware Abstraction:** The code acts as a thin wrapper over AVX-512 intrinsics, effectively treating the CPU as a custom ASIC. The reliance on `alignas(64)` and `__restrict` pointers ensures the compiler generates optimal, non-aliased load instructions, maximizing the utilization of the L1 cache.

---

## File: `nn\core\execution\importance.cpp`

### Architectural Analysis: `importance.cpp`

This module serves as the **Heuristic Gating Layer** for the CENTAUR Neural Engine. It performs real-time signal analysis to determine if an input vector warrants high-precision compute cycles or memory-intensive learning updates.

#### 1. Core Purpose
The `ImportanceClassifier` acts as a **dynamic scheduler**. By quantifying information density (Saliency) and contextual divergence (Novelty), it prevents the engine from wasting cycles on redundant or low-entropy data. It transforms raw input streams into binary decision flags (`is_fact`, `should_learn`) that dictate the execution path of the downstream pipeline.

#### 2. Critical Function: `classify()`
The function implements a two-pass statistical reduction on the `d_model` vector:
*   **Saliency (Variance):** Computes the second moment of the input vector. This acts as a high-pass filter; low-variance signals (noise/flat-line) are discarded, preserving compute for high-entropy features.
*   **Novelty (Cosine Similarity):** Measures the angular distance between the current input and the latent `state`. This identifies "surprise" in the data stream, triggering learning only when the input deviates significantly from the established model state.

#### 3. AVX-512 & Physics Architecture Integration
To maintain the **zero-cost, heapless** requirement, this implementation is designed for immediate refactoring into AVX-512 intrinsics:

*   **Vectorization Potential:** The current scalar loops are prime candidates for `_mm512_loadu_ps` and `_mm512_fmadd_ps`. The variance calculation can be reduced using `_mm512_reduce_add_ps`, eliminating the need for horizontal accumulation overhead.
*   **Heapless Execution:** The function operates entirely on stack-allocated `ImportanceDecision` structs and pointer-based memory access. By avoiding dynamic allocation, it ensures deterministic latency—a prerequisite for the CENTAUR engine's cycle-accurate execution model.
*   **Physics-Informed Gating:** The `act_cycles` logic (2 vs 1) directly maps to the engine's **Variable-Rate Execution (VRE)**. By modulating the cycle count based on `is_fact`, the engine effectively implements a "compute-on-demand" physics model, where energy consumption is proportional to the information density of the input.

#### 4. Architectural Impact
*   **Throughput Optimization:** By filtering inputs at the entry point, the engine avoids unnecessary tensor operations, effectively increasing the "effective" FLOPS per watt.
*   **State Stability:** The `should_learn` threshold prevents the model from overfitting on noise, ensuring that the latent `state` only updates when the input provides a statistically significant delta.
*   **Deterministic Pipeline:** The absence of branching logic inside the hot loops (once vectorized) ensures a constant-time execution path, critical for real-time inference stability.

---

## File: `nn\core\execution\importance.hpp`

### Architectural Analysis: `importance.hpp`

The `ImportanceClassifier` serves as the **gating heuristic engine** for the CENTAUR Neural Engine. It acts as a high-throughput filter that determines the cognitive relevance of incoming data streams before they reach the memory-write pipeline.

#### Core Purpose
This component implements a **dynamic attention-gating mechanism**. By quantifying information entropy, novelty, and surprise, it prevents "memory pollution" by ensuring only high-value state transitions trigger expensive write operations. It effectively transforms the neural network from a static compute graph into a selective, event-driven system.

#### Critical Components
*   **`ImportanceDecision` Struct**: A packed POD structure designed for register-level alignment. It encapsulates the heuristic state required for downstream decision-making.
*   **`classify()`**: The primary entry point. Architecturally, this function is intended to be implemented as a **SIMD-vectorized kernel**. It consumes three contiguous memory buffers (`x`, `state`, `prediction`) to compute the decision vector.

#### AVX-512 & Zero-Cost Integration
To maintain the "heapless" and "zero-cost" requirements of the CENTAUR architecture, this module adheres to the following constraints:

1.  **Register-Bound Execution**: The `classify` function is designed to operate on `__m512` registers. By processing 16 `float` elements per iteration, it minimizes cache pressure and avoids heap allocation by utilizing stack-allocated or pre-allocated scratchpads.
2.  **Branchless Heuristics**: The logic within `classify` must be implemented using AVX-512 mask registers (`k0`-`k7`) rather than conditional branching. This ensures the pipeline remains saturated, preventing branch misprediction penalties during high-frequency inference.
3.  **Memory Locality**: By accepting raw pointers (`const float*`), the interface supports **Zero-Copy data access**. This allows the classifier to operate directly on the output of the preceding layer's buffer, eliminating the need for intermediate data movement or serialization.
4.  **Deterministic Latency**: The `act_cycles` field enables the engine to adjust its dynamic depth based on the `saliency` score. This allows the system to "short-circuit" computation for low-importance inputs, directly contributing to the engine's power-efficiency and real-time performance targets.

---

## File: `nn\core\execution\multimodal_engine.cpp`

### Architectural Analysis: CENTAUR Multimodal Engine

The `MultimodalEngine` serves as the high-throughput inference orchestrator for the CENTAUR architecture. It replaces traditional dense matrix multiplication (GEMM) with a **Geometric Schema**—a sparse, spectral-domain routing mechanism that treats the model state as a physical wavefront rather than a static tensor.

#### 1. Core Architectural Philosophy
*   **Spectral-Geometric Duality:** By performing operations in the Walsh-Hadamard domain (`fwht`/`ifwht`), the engine achieves global information mixing with $O(D \log D)$ complexity, bypassing the $O(D^2)$ bottleneck of standard Transformer layers.
*   **Binary Logic Routing (BCT):** The `step_geometric` function implements a "transistor-level" weight routing system. Instead of floating-point MACs, it uses `__mmask16` to gate the wavefront, effectively treating the weight matrix as a programmable logic array (PLA).
*   **Heapless Execution:** The engine relies on pre-allocated `SiliconWavefront` buffers and stack-aligned scratchpads (`alignas(64)`), ensuring deterministic latency and cache-locality essential for real-time neural physics.

#### 2. Critical Execution Path
*   **`step_swarm` (Recursive Thought Cycle):** Implements an adaptive inference loop. It uses a "Silicon-ACT" (Adaptive Computation Time) halting mechanism, where the swarm terminates early if the accumulated confidence (`accumulated_h`) exceeds the threshold, minimizing cycles per inference.
*   **`step_geometric` (The Inference Kernel):**
    *   **Ingestion:** Uses AVX-512 `_mm512_fmadd_ps` for gated fusion of multimodal inputs into the primary wavefront.
    *   **GLR Recurrence:** Applies Gated Linear Recurrence (GLR) to maintain temporal state persistence without explicit hidden-state buffers.
    *   **Geometric Hop:** Invokes the `WavefrontRouter` to perform non-linear, stochastic state transitions, simulating a physical system traversing a graph.
    *   **BCT Routing:** The core logic gate implementation. It uses `_mm512_maskz_loadu_ps` to perform conditional accumulation, effectively performing sparse matrix-vector multiplication (SpMV) via bitmasking.
    *   **RMS Actuation:** Normalizes the wavefront energy using a horizontal reduction (`_mm512_reduce_add_ps`) to ensure numerical stability across recursive cycles.

#### 3. Performance & Systems Optimization
*   **SIMD Vectorization:** The engine is strictly bound to 512-bit registers. By aligning all buffers to 64-byte boundaries, it ensures optimal cache-line utilization and prevents split-load penalties.
*   **Zero-Cost Abstraction:** The use of `std::unique_ptr` for registry management and `alignas` stack arrays ensures that the engine maintains a fixed memory footprint, avoiding runtime heap fragmentation during the swarm cycle.
*   **Spectral In-Place Processing:** By performing `fwht` and `ifwht` in-place, the engine minimizes memory bandwidth pressure, keeping the wavefront state resident in L1/L2 cache throughout the entire geometric transformation pipeline.

#### 4. Summary of Contribution
This engine transforms the neural network from a static weight-lookup table into a **dynamic physical simulation**. By replacing dense weights with binary masks and spectral transforms, it achieves a significant reduction in FLOPs while maintaining high-dimensional representational capacity, perfectly aligning with the CENTAUR goal of high-efficiency, low-latency neural execution.

---

## File: `nn\core\execution\multimodal_engine.hpp`

### Architectural Analysis: `multimodal_engine.hpp`

The `MultimodalEngine` serves as the high-level orchestrator for the CENTAUR Neural Engine, transitioning from traditional VNNI-based matrix multiplication to a **Geometric Schema Inference** paradigm. It leverages AVX-512 to maintain a "heapless" execution flow by mapping neural states directly onto pre-allocated silicon-aligned memory buffers.

#### Core Architectural Pillars
*   **Geometric Schema Inference:** Bypasses standard dense-layer compute by treating weights as structural pointers. This reduces latency by performing direct memory-mapped transformations, effectively treating the neural network as a geometric manifold rather than a series of dot products.
*   **Silicon Swarm (Chained Recurrence):** Implements a zero-copy feedback loop where the output of agent $N$ is injected into the register-file of agent $N+1$. This minimizes cache-miss penalties by keeping the wavefront state resident in L1/L2 cache across sequential inference steps.
*   **Wavefront Routing:** The `WavefrontRouter` acts as the hardware abstraction layer, managing the spatial distribution of the `SiliconWavefront` across AVX-512 lanes.

#### Critical Function Breakdown

| Function | Architectural Role |
| :--- | :--- |
| `step_geometric` | Executes the primary inference path. By bypassing VNNI, it utilizes direct pointer arithmetic to perform structural transformations, minimizing instruction overhead. |
| `step_swarm` | Manages the recursive state propagation. It enforces a strict memory-ordering constraint to ensure the output of one agent is immediately available for the next without synchronization barriers. |
| `get_saliency_heatmap` | Performs a destructive read of the `SiliconWavefront`. It collapses high-dimensional latent space into a 16x16 grid, optimized for AVX-512 `vmovups` operations to minimize bus contention. |
| `get_weight_registry` | Exposes raw pointers to the `WeightRegistry`. This allows the `WeightAdapter` to perform in-place weight updates (online learning) without reallocating the underlying memory buffers. |

#### Zero-Cost/Heapless Integration
*   **Memory Alignment:** The engine relies on `SiliconWeights` and `SiliconWavefront` types, which are designed to be cache-line aligned (64-byte) to facilitate optimal AVX-512 load/store throughput.
*   **Deterministic Execution:** By utilizing `std::unique_ptr` for fixed-size buffers and avoiding dynamic resizing during the `step_*` functions, the engine eliminates non-deterministic heap allocations during the inference hot-path.
*   **Spectral Coupling:** The inclusion of `KroneckerRLSState` suggests that the engine performs recursive least squares updates in the spectral domain, allowing for real-time model adaptation without the overhead of backpropagation through the entire graph.

---

## File: `nn\core\execution\route_planner.cpp`

### Architectural Analysis: `route_planner.cpp`

The `route_planner` serves as the **dynamic scheduling gatekeeper** for the CENTAUR Neural Engine. It transforms raw activation metrics into a compact, cache-aligned index map, enabling the execution engine to bypass inactive compute nodes (sparsity-aware execution).

#### 1. Core Purpose
The module implements **Sparsity-Aware Token Routing**. By filtering or selecting high-importance tokens before the main compute kernels, it minimizes memory bandwidth pressure and maximizes SIMD utilization by ensuring the execution pipeline only processes "hot" data.

#### 2. Critical Function Breakdown

*   **`shuffle_active_tokens`**: Acts as a **gather-scatter pre-processor**. It performs a non-contiguous to contiguous memory transformation. By utilizing `_mm512_loadu_ps` and `_mm512_storeu_ps`, it maximizes throughput for the subsequent compute kernels, ensuring that the data fed into the AVX-512 units is perfectly aligned and contiguous, eliminating the need for expensive gather operations inside the compute-heavy kernels.
*   **`plan_route_threshold`**: Implements **SIMD-parallel predicate filtering**. It leverages `_mm512_cmp_ps_mask` to evaluate 16 tokens per cycle. The use of `_mm512_cmp_ps_mask` combined with `_BitScanForward` (or `__builtin_ctz`) allows for branchless identification of active indices, effectively turning a conditional filtering task into a high-throughput bit-manipulation stream.
*   **`plan_route_topk`**: Provides **global importance ranking**. While `std::nth_element` is used for $O(N)$ selection, it acts as the "slow path" for non-threshold-based routing. It ensures that the `active_indices` buffer is sorted, which is critical for maintaining deterministic memory access patterns in downstream hardware-accelerated kernels.

#### 3. Contribution to "Zero-Cost, Heapless" Architecture

*   **Memory Determinism**: The `RoutePlan` object encapsulates a pre-allocated, 64-byte aligned buffer. By managing this memory via `_aligned_malloc` and `std::unique_ptr` (or raw pointer management), the engine avoids runtime heap allocations during the inference loop, preventing non-deterministic latency spikes.
*   **SIMD-First Design**: The architecture forces data into a format that aligns with the 512-bit register width. By filtering indices into a contiguous array, it allows the execution engine to treat the neural network as a dense matrix operation, even when the underlying data is sparse.
*   **Branch Minimization**: The use of `[[likely]]` attributes and mask-based logic in `plan_route_threshold` minimizes pipeline stalls. The logic is designed to keep the CPU front-end saturated, treating the routing process as a streaming data task rather than a control-flow task.

#### 4. Architectural Critique
*   **Bottleneck**: `plan_route_topk` currently relies on `std::vector` allocation. To achieve true "heapless" status, this should be refactored to use a stack-allocated buffer or a pre-allocated scratchpad memory pool provided by the `ExecutionEngine` context.
*   **Optimization Opportunity**: The `shuffle_active_tokens` function could be further optimized using `_mm512_mask_loadu_ps` if the sparsity pattern is known, potentially allowing for a single-pass load-and-shuffle that avoids the intermediate `p_src` pointer arithmetic.

---

## File: `nn\core\execution\route_planner.hpp`

### Architectural Analysis: `route_planner.hpp`

The `route_planner.hpp` module serves as the **Data Orchestration Layer** for the CENTAUR Neural Engine. Its primary objective is to mitigate the "Gather Penalty" inherent in sparse neural execution by transforming non-contiguous memory access patterns into linear, cache-friendly streams suitable for AVX-512 vectorization.

#### Core Purpose
This component implements a **Dynamic Routing Fabric**. By decoupling token selection (thresholding/Top-K) from execution, it ensures that the subsequent compute kernels operate on packed, contiguous memory blocks. This maximizes L1/L2 cache hit rates and enables the use of `vmovups` and `vgatherdps` (where necessary) with optimal alignment, effectively turning sparse neural workloads into dense, SIMD-saturated pipelines.

#### Critical Functions

*   **`shuffle_active_tokens`**: The engine's "Data Compactor." It performs a gather-to-contiguous-stream transformation. By reordering sparse activations into a dense buffer, it allows the AVX-512 execution units to operate at peak throughput without stalling on cache misses or non-aligned memory boundaries.
*   **`plan_route_threshold`**: A high-speed predicate filter. It identifies active neural pathways based on activation magnitude. It is designed to be branch-prediction friendly, facilitating the creation of a compact index map for the `shuffle` phase.
*   **`plan_route_topk`**: A selection primitive that identifies the most significant activations. It acts as the gatekeeper for dynamic sparsity, ensuring only the most relevant data enters the compute pipeline, thereby minimizing total FLOPs.

#### Integration with AVX-512 Physics Architecture

*   **Zero-Cost Memory Management**: The `aligned_unique_ptr` and `AlignedDeleter` ensure that all buffers are 64-byte aligned. This is non-negotiable for AVX-512, as it prevents split-load penalties and allows for the use of aligned load/store instructions (`vmovaps`), which are essential for saturating the 512-bit wide data paths.
*   **Heapless Execution Path**: While `RoutePlan` uses `unique_ptr` for initial allocation, the architecture is designed to be "heapless" during the hot-path execution. By pre-allocating `max_capacity` buffers, the engine avoids runtime allocations during inference, ensuring deterministic latency and preventing fragmentation.
*   **SIMD-Ready Data Layout**: The `__restrict` qualifiers are critical here; they inform the compiler that the source and destination buffers do not alias, enabling aggressive loop unrolling and vectorization of the shuffle logic. This ensures the "physics" of the data movement matches the hardware's capability to process 16 single-precision floats per cycle.

**Architectural Verdict:** This module is the bridge between high-level neural logic and low-level hardware efficiency. It enforces the "Contiguous-First" design philosophy required to keep the CENTAUR execution units fed at the theoretical limit of the AVX-512 ISA.

---

## File: `nn\core\execution\silicon_automation.cpp`

### Architectural Analysis: `silicon_automation.cpp`

This module serves as the **meta-programming orchestration layer** for the CENTAUR Neural Engine. Its primary role is to enforce strict codebase hygiene and hardware-level optimization constraints by automating the removal of non-deterministic runtime overhead (telemetry, IDE bloat, and dynamic configuration) that would otherwise pollute the instruction cache and interfere with AVX-512 vectorization pipelines.

#### Core Purpose
To maintain a "Zero-Cost" execution environment by stripping high-level abstraction overhead from the source tree before compilation. By automating the "Silicon Wipe," it ensures that the binary remains lean, heapless, and strictly deterministic, preventing branch mispredictions caused by unnecessary telemetry or configuration-checking logic.

#### Critical Functions

*   **`batch_refactor`**: Acts as the atomic transformation primitive. It provides the timing telemetry necessary to track the cost of refactoring operations, ensuring that the automation process itself does not introduce latency into the CI/CD pipeline.
*   **`walk_and_replace`**: Implements a recursive filesystem traversal with robust exception handling. It enforces a strict whitelist of file extensions, preventing the accidental corruption of binary blobs or non-source assets while performing in-place string substitution.
*   **`silicon_wipe`**: The primary architectural enforcement mechanism. It targets specific "bloat" patterns (telemetry, sync, and auto-update hooks) and replaces them with static, compile-time constants. This effectively "hard-wires" the engine to a production-only state, eliminating runtime branching logic that would otherwise break the predictability required for AVX-512 SIMD throughput.
*   **`trigger_fast_build`**: A hook for the build system integration. It is designed to trigger a saturated build process, ensuring that the refactored code is immediately compiled into the final instruction stream, maintaining the integrity of the "Freedom Refactor."

#### Contribution to AVX-512 Physics Architecture
The CENTAUR engine relies on **deterministic execution paths** to maximize the utilization of 512-bit wide registers. By automating the removal of telemetry and dynamic configuration via `silicon_wipe`, this module:
1.  **Reduces Instruction Cache Pressure**: Eliminates conditional branches that would otherwise cause pipeline stalls.
2.  **Enforces Heapless Constraints**: By disabling high-level features that rely on dynamic memory allocation, it ensures the engine remains within the L1/L2 cache footprint.
3.  **Maximizes SIMD Throughput**: By ensuring the codebase is stripped of non-essential logic, it allows the compiler to perform aggressive loop unrolling and vectorization without the risk of hidden side-effects or branch-heavy code paths.

---

## File: `nn\core\execution\silicon_automation.hpp`

### Architectural Analysis: `silicon_automation.hpp`

This component serves as the **Meta-Programming Orchestrator** for the CENTAUR Neural Engine. It shifts infrastructure management from interpreted Python overhead to a compiled, memory-mapped C++ execution layer, ensuring the development toolchain matches the performance profile of the target AVX-512 silicon.

#### Core Purpose
To eliminate I/O bottlenecks and latency in the CI/CD pipeline by leveraging direct filesystem memory-mapping and SIMD-accelerated string processing. By bypassing the Python interpreter, it enables "zero-cost" refactoring and build orchestration that operates at the speed of the underlying hardware bus.

#### Critical Function Breakdown

*   **`batch_refactor`**: Implements high-throughput text transformation. By utilizing `std::string_view` and memory-mapped files, it avoids heap allocations during pattern matching. In an AVX-512 context, this should be implemented using `_mm512_cmpeq_epi8_mask` to perform parallel character comparisons, enabling multi-gigabyte-per-second search-and-replace throughput.
*   **`trigger_fast_build`**: Acts as a low-latency interface to the build system. It bypasses shell-out overhead by invoking build primitives directly via system calls, minimizing context switching and maintaining cache locality for the compiler's hot paths.
*   **`silicon_wipe`**: A destructive cleanup utility designed for deterministic state management. It performs bulk filesystem unlinking, ensuring the workspace remains "heapless" and free of stale artifacts that could pollute the instruction cache or build environment.

#### Contribution to AVX-512 Physics Architecture
*   **Zero-Cost Abstraction**: By replacing Python, the automation layer eliminates the non-deterministic garbage collection and interpreter overhead that typically plagues neural engine toolchains.
*   **Heapless Execution**: The architecture favors stack-allocated buffers and pre-allocated memory pools for file operations, preventing fragmentation during large-scale refactoring sweeps.
*   **SIMD Alignment**: The design philosophy assumes that if the automation engine is as performant as the neural engine itself, the "compile-test-deploy" loop becomes a constant-time operation, effectively treating the entire codebase as a data-parallel stream ready for AVX-512 vectorization.

---

## File: `nn\core\execution\silicon_memory.cpp`

### Architectural Analysis: `silicon_memory.cpp`

This module serves as the **Data Layout Orchestrator** for the CENTAUR Neural Engine. It enforces strict memory alignment and deterministic initialization patterns required for AVX-512 vectorization pipelines.

#### 1. Core Purpose
The file manages the lifecycle of weight tensors and wavefront state buffers. By utilizing `nca::simd::make_aligned_unique`, it ensures all memory allocations are cache-line aligned (typically 64-byte boundaries), a prerequisite for `vmovaps` and `vmovups` instructions to avoid performance penalties or alignment faults during high-throughput SIMD execution.

#### 2. Critical Function Breakdown

*   **`initialize_unit_noise`**: Establishes the stochastic baseline for the `vision_A` tensor (a 524,288-element block) and GLR (Geometric Learning Rate) parameters. The use of a fixed seed (`42`) ensures reproducibility in the silicon-emulated environment, critical for debugging non-deterministic convergence in the neural engine.
*   **`initialize_binary_curve`**: Implements a bit-packed mask architecture. By storing masks as `uint16_t` arrays, it optimizes for AVX-512 `vpmovm2w` or `vpbroadcastw` operations, allowing the engine to apply binary gating or sparsity masks across 16-bit lanes with minimal instruction overhead.
*   **`SiliconWavefront` Constructor/Reset**: Manages the transient state of the engine. The use of `std::memset` for zero-initialization is a performance-critical path; it leverages highly optimized platform-specific implementations to clear the `d_model` buffers, ensuring the engine starts from a clean state without heap fragmentation.

#### 3. Contribution to "Zero-Cost, Heapless" Architecture
While the code currently uses `make_aligned_unique` (which invokes the heap), it acts as a **static-allocation proxy**. In the production CENTAUR pipeline, these calls are intended to be replaced by a custom `ArenaAllocator` or `StaticMemoryPool`. 

*   **Vectorization Alignment**: By enforcing alignment at the point of allocation, the engine eliminates the need for runtime pointer adjustment or unaligned load/store instructions, effectively achieving "zero-cost" memory access.
*   **Cache Locality**: The contiguous layout of `SiliconWavefront` buffers ensures that when the AVX-512 unit fetches the `state` or `momentum` vectors, it maximizes L1/L2 cache line utilization, minimizing stalls during the compute-heavy backpropagation or inference cycles.
*   **Geometric Schema Migration**: The removal of legacy MoE dependencies indicates a shift toward a flatter, more predictable memory topology, reducing pointer chasing and branch mispredictions in the execution hot-path.

---

## File: `nn\core\execution\silicon_memory.hpp`

### Architectural Analysis: `silicon_memory.hpp`

This header defines the memory layout for the **CENTAUR Neural Engine’s** execution context. It enforces strict alignment and cache-line-aware data structures to facilitate direct AVX-512 vectorization without runtime overhead.

#### 1. `SiliconWeights` (Static Kernel Registry)
Acts as the global weight-space container. By utilizing `aligned_unique_ptr`, it ensures that all weight tensors are pinned to 64-byte boundaries, satisfying the alignment requirements for `vmovaps` and `vmovups` instructions.
*   **Vision Primitives (`vision_A/B/C`):** Pre-allocated buffers for high-throughput matrix-vector operations. These are designed for direct mapping into ZMM registers, minimizing load-to-use latency.
*   **Binary Curve Tree (BCT):** A quantized weight representation. By storing masks as `uint16_t` and applying a scalar `bct_scale/offset`, the engine performs dequantization on-the-fly within the AVX-512 pipeline, effectively doubling effective memory bandwidth for transistor-level weight operations.

#### 2. `SiliconWavefront` (Transient Execution State)
Represents the "mental state" of an agent. It is designed for **zero-cost state transitions** by maintaining fixed-size buffers that map 1:1 to the engine's internal recurrence logic.
*   **`state` / `momentum`:** These buffers are sized to `d_model` and are intended to be processed via Fused Multiply-Add (FMA) instructions.
*   **`h_glr`:** Dedicated memory for the Gated Linear Recurrence (GLR) hidden state, ensuring that recurrence updates do not contend with the primary prediction buffer.

#### 3. Architectural Contribution to "Heapless" Physics
*   **Deterministic Allocation:** By centralizing memory in `SiliconWeights` and `SiliconWavefront`, the engine avoids dynamic heap fragmentation during the inference loop. All memory is allocated at initialization (or session start), ensuring the hot path is strictly pointer-arithmetic based.
*   **Cache-Line Alignment:** The use of `nca::simd::aligned_unique_ptr` prevents "split-loads" where a single AVX-512 vector spans two cache lines, which would otherwise trigger a performance penalty on the memory controller.
*   **Zero-Copy Semantics:** The structure allows the execution engine to pass pointers directly to the AVX-512 kernels. Because the memory is pre-aligned and pre-allocated, the "physics" of the neural engine (the mathematical transformations) can operate directly on the memory-mapped weights without intermediate buffering or serialization.

#### 4. Critical Functions
*   **`initialize_unit_noise`:** Pre-populates the weight space with stochastic initialization parameters, ensuring that the engine is ready for immediate execution without requiring a cold-start phase.
*   **`initialize_binary_curve`:** Configures the BCT lookup tables. This function is the gateway for the engine's ability to perform high-density inference by mapping low-precision masks to high-precision floating-point outputs via SIMD-accelerated expansion.

---

## File: `nn\core\execution\wavefront_router.cpp`

### Architectural Analysis: `WavefrontRouter`

The `WavefrontRouter` acts as the high-throughput interconnect fabric for the CENTAUR Neural Engine. It maps sparse concept activations to downstream targets using a **SIMD-parallelized scatter-gather pattern**, optimized for AVX-512 execution units.

#### Core Purpose
It transforms a graph-based neural topology into a contiguous, cache-aligned **Structure of Arrays (SoA)** format. By enforcing a fixed `wavefront_width=16`, it ensures that every concept transition maps perfectly to a single `zmm` register, enabling deterministic, branchless routing of activation energy.

#### Critical Functions
*   **`load_geometric_graph`**: Performs the critical transformation from a high-level graph representation to a flattened, SIMD-friendly memory layout. It enforces padding to 16-lane boundaries, ensuring that subsequent AVX-512 loads are always aligned to 64-byte cache lines, eliminating unaligned access penalties.
*   **`step_wavefront`**: The engine's hot path. It performs a **SIMD-fused multiply-add (FMA)** to apply stochastic temperature-based noise to transition probabilities. By utilizing `_mm512_fmadd_ps`, it integrates exploration noise directly into the routing logic without additional scalar overhead.

#### Architectural Contribution
*   **Zero-Cost Abstraction**: The router minimizes pointer chasing by using `graph_offsets_` to index into a flat memory space. This allows the CPU to prefetch transition data linearly, maximizing the efficiency of the L1/L2 cache hierarchy.
*   **Heapless/Deterministic Execution**: While the current implementation uses `make_aligned_unique` (which invokes the heap), the architecture is designed for **static allocation**. In a production CENTAUR deployment, these buffers are pre-allocated during the initialization phase, ensuring the `step_wavefront` loop operates on pre-warmed, pinned memory regions.
*   **SIMD-Parallel Stochasticity**: By generating noise at the register level, the router avoids the bottleneck of scalar RNG calls. The use of `_mm512_set1_ps` and `_mm512_mul_ps` ensures that the entire wavefront of 16 potential concept transitions is processed in a single clock cycle, maintaining the "pure C++" performance profile required for real-time neural physics.

#### Critical Bottleneck Note
The current `step_wavefront` implementation contains a scalar `for` loop for the final scatter operation (`next_state[target_id] += out_amps[lane]`). To achieve true "zero-cost" performance, this should be refactored to use **AVX-512 scatter instructions** (`_mm512_i32scatter_ps`) to eliminate the scalar loop and allow the hardware to handle memory contention at the cache-line level.

---

## File: `nn\core\execution\wavefront_router.hpp`

### Architectural Analysis: `wavefront_router.hpp`

The `WavefrontRouter` serves as the **stochastic execution engine** for the CENTAUR Neural Engine. It replaces traditional dense matrix-vector multiplication (GEMM) with **Geometric Pointer Chasing**, mapping neural activations to sparse, graph-based structural transitions.

#### 1. Core Purpose: Geometric Execution
Instead of computing weights, the engine traverses a pre-compiled `GeometricBranch` graph. This shifts the computational burden from floating-point arithmetic to **memory-latency-bound gather operations**, effectively treating neural inference as a parallel path-finding problem across a high-dimensional manifold.

#### 2. Critical Components
*   **`GeometricBranch` (8-byte Schema):** A cache-aligned, packed structure designed to fit exactly 8 entries per 64-byte cache line. This minimizes cache misses during the `_mm512_i32gather_epi32` phase, ensuring that each SIMD lane fetches its next state transition in a single cycle.
*   **`SimdRandomState` (Xorshift-based):** A vectorized PRNG implementing a 16-lane Xorshift algorithm. By generating 16 uniform floats per cycle, it enables stochastic branching (temperature-based sampling) without branching divergence, maintaining full AVX-512 throughput.
*   **SoA Flattening:** The transition from `std::vector<GeometricBranch>` to `flat_pointers_` and `flat_probs_` is the architectural pivot. By decoupling the structural pointers from the probability weights, the engine enables **SIMD-width-aligned gathers**, bypassing the overhead of pointer chasing in scalar code.

#### 3. Execution Mechanics
*   **`step_wavefront`:** The primary execution loop. It utilizes the `rng_` to sample against the `flat_probs_` while simultaneously performing a gather on `flat_pointers_`. This effectively "walks" 16 parallel realities through the graph simultaneously.
*   **Zero-Cost/Heapless Strategy:** The use of `aligned_unique_ptr` and flattened arrays ensures that the graph resides in a contiguous, cache-friendly memory block. This eliminates pointer-chasing latency and heap fragmentation, allowing the engine to operate as a deterministic state machine once the graph is loaded.

#### 4. Architectural Contribution
This module transforms the neural network into a **deterministic geometric traversal**. By replacing dense weights with structural pointers, the CENTAUR engine achieves:
1.  **Instruction-Level Parallelism:** Eliminates branch mispredictions by using mask-based selection (`rule_mask`) rather than conditional logic.
2.  **Memory Throughput:** Maximizes AVX-512 bandwidth by aligning graph data to 64-byte boundaries, ensuring that every load instruction saturates the L1 cache.
3.  **Stochastic Stability:** The `temperature` parameter allows for controlled exploration of the graph, enabling the engine to perform probabilistic inference without the overhead of traditional softmax layers.

---

## File: `nn\core\layers\glr.cpp`

### Architectural Analysis: `glr.cpp`

The `glr.cpp` module implements a **Gated Linear Recurrence (GLR)** backbone, a compute-bound primitive essential for state-space models (SSMs). It prioritizes cache-locality and instruction-level parallelism (ILP) to minimize memory stall cycles on the CENTAUR Neural Engine.

#### 1. Core Purpose
The layer performs a fused multiply-add operation: $h_t = \alpha_t \odot h_{t-1} + \beta_t \odot x_t$. By fusing the gating ($\alpha, \beta$) and the recurrence into a single pass, it maximizes the arithmetic intensity of the AVX-512 FMA units, effectively turning a memory-bound operation into a compute-bound one.

#### 2. Critical Architectural Components
*   **`CachePolicy` Integration**: Uses template metaprogramming to inject prefetch distances and memory strategies at compile-time. This eliminates runtime branching for cache management, ensuring the instruction pipeline remains saturated.
*   **Loop Unrolling (4x)**: The `glr_step_avx512` function unrolls the loop to process 256 bytes (4x 512-bit registers) per iteration. This depth is specifically tuned to hide the latency of the FMA pipeline and maximize the throughput of the load/store units.
*   **Masked Tail Handling**: Employs AVX-512 opmask registers (`__mmask16`) to handle non-aligned `d_size` inputs. This avoids the traditional "scalar cleanup loop," maintaining branchless execution and preventing pipeline flushes.
*   **`NCA_DISPATCH_KERNEL`**: A zero-cost abstraction layer that selects the optimal ISA path (AVX-512 vs. AVX2 vs. Scalar) at runtime based on CPUID, ensuring the engine remains portable without sacrificing performance on target hardware.

#### 3. Contribution to "Zero-Cost, Heapless" Architecture
*   **Memory Determinism**: By operating on raw pointers (`__restrict`) and avoiding dynamic allocations, the layer ensures zero heap fragmentation and predictable cache behavior.
*   **Zero-Overhead Abstraction**: The use of `constexpr` and `[[likely]]` attributes allows the compiler to inline the logic directly into the calling graph, effectively removing the function call overhead.
*   **Hardware-Aligned Prefetching**: The prefetch logic is explicitly tied to the cache line size (64 bytes), preventing cache pollution and ensuring that the L1/L2 hierarchy is primed exactly when the data is required by the FMA units.

#### 4. Performance Criticality
The implementation assumes a **streaming data model**. By utilizing `_mm512_loadu_ps` and `_mm512_storeu_ps` with explicit prefetching, it minimizes the penalty of unaligned memory access, which is common in neural network tensors. The `_mm_sfence()` call (conditional on `use_nt_stores`) ensures memory consistency for non-temporal stores, critical for maintaining state integrity in high-throughput inference pipelines.

---

## File: `nn\core\layers\glr.hpp`

### Architectural Analysis: `glr.hpp`

The `glr_step` function serves as the fundamental recurrence primitive for the Gated Linear RNN (GLR) within the CENTAUR engine. It implements a linear state-space transition optimized for high-throughput temporal processing.

#### Core Purpose
*   **State-Space Evolution:** Implements the first-order linear recurrence $h_t = \alpha \odot h_{t-1} + \beta \odot x_t$.
*   **Temporal Dependency:** Facilitates sequential state updates while maintaining strict memory locality to minimize cache misses during long-sequence inference.

#### AVX-512 Implementation Strategy
To achieve the "zero-cost" objective, the implementation must leverage the following architectural patterns:

*   **Vectorized Fused Multiply-Add (FMA):** The operation maps directly to `_mm512_fmadd_ps`. By processing 16 `float` elements per instruction, the engine saturates the execution ports, effectively hiding the latency of the dependency chain.
*   **Memory Alignment & Streaming:** The function assumes `h`, `alpha`, `beta`, and `x` are 64-byte aligned. This allows for `_mm512_load_ps` and `_mm512_store_ps` (or non-temporal `_mm512_stream_ps` if the state is not immediately reused), bypassing the L1 cache bottleneck.
*   **Heapless Execution:** By operating on raw pointers provided by the caller (typically pre-allocated static buffers or stack-allocated scratchpads), the layer avoids dynamic memory allocation, ensuring deterministic execution time and zero fragmentation.

#### Integration within CENTAUR
*   **Zero-Cost Abstraction:** The header-only design and lack of virtual dispatch ensure the compiler can inline the recurrence directly into the execution graph, enabling inter-procedural optimization (IPO) across layer boundaries.
*   **Physics-Informed Latency:** By maintaining a pure C++ interface, the engine allows the compiler to perform loop unrolling and software pipelining, critical for maintaining the sub-microsecond latency requirements of the CENTAUR physics-based neural architecture.
*   **Data Layout:** The design favors a "Structure of Arrays" (SoA) approach, ensuring that the `d_size` dimension is contiguous, which is essential for maximizing the 512-bit wide SIMD lanes.

---

## File: `nn\core\layers\halting.cpp`

### Architectural Analysis: `halting.cpp`

The `halting.cpp` module implements the **Adaptive Computation Time (ACT)** mechanism for the CENTAUR Neural Engine. It serves as the temporal termination logic for recurrent neural states, enabling dynamic depth per input sample.

#### 1. Core Purpose: Dynamic Temporal Termination
The module manages the "Halting Gate," which regulates the iterative refinement of hidden states. By calculating the accumulation of probability mass (`p_sum`) and the residual influence (`remainder`), it allows the engine to bypass unnecessary compute cycles once the halting criterion (`ex > 0.5f`) is met. This is critical for maintaining constant-time performance bounds in variable-depth inference.

#### 2. AVX-512 Implementation Strategy
To maintain the **zero-cost, heapless** requirement, the logic is designed for direct mapping to `__m512` registers:
*   **Fused Multiply-Add (FMA) Utilization:** The update rule `pt = ex * pc + (1.f - ex) * pt` is architected to map directly to `_mm512_fmadd_ps`, minimizing instruction latency and register pressure.
*   **Branchless Predication:** The `should_halt` logic is intended to be implemented via `_mm512_cmp_ps_mask`, allowing the engine to mask out inactive lanes in a SIMD vector without branching, preserving the pipeline flow.
*   **State Encapsulation:** The `state` structure is designed for stack-allocation or pre-allocated buffer residency, ensuring zero heap allocation during the inference loop.

#### 3. Integration with CENTAUR Physics Architecture
*   **Deterministic Latency:** By enforcing a fixed-size `state` object, the engine avoids non-deterministic memory management overhead, ensuring the execution time is strictly a function of the input vector size and the halting threshold.
*   **SIMD Alignment:** The logic assumes input tensors are aligned to 64-byte boundaries, allowing the halting gate to process 16 FP32 hidden state components per cycle.
*   **Zero-Copy State Updates:** The `state.p_sum` and `state.remainder` updates are performed in-place within the register file, preventing cache-line thrashing and maintaining high L1/L2 cache locality.

#### 4. Critical Architectural Constraints
*   **Register Pressure:** The implementation must balance the `pt` (previous state), `pc` (current state), and `ex` (exit probability) registers. With 32 ZMM registers available, the architecture supports unrolling the halting gate across multiple channels to maximize throughput.
*   **Precision Handling:** The use of `1.f - ex` requires careful handling of floating-point rounding modes to ensure that the `p_sum` converges to 1.0, preventing numerical drift in long-running recurrent sequences.

---

## File: `nn\core\layers\halting.hpp`

### Architectural Analysis: `halting.hpp`

The `Halting Gate` implements **Adaptive Computation Time (ACT)**, a dynamic control-flow mechanism that transforms fixed-depth neural networks into variable-depth inference engines. By predicting a halting probability $p_t$ at each layer, the engine prunes redundant compute cycles for "easy" inputs, directly optimizing the **FLOPs-per-inference** metric.

#### 1. Core Purpose
*   **Dynamic Depth Scaling:** Enables early-exit logic based on internal state entropy.
*   **Compute Budgeting:** Shifts execution resources from high-entropy (complex) regions to low-entropy (simple) regions, effectively flattening the latency distribution across heterogeneous input sets.

#### 2. Critical Components
*   **`HaltingState` (Stateful Accumulator):**
    *   Maintains the running probability sum (`p_sum`) and the residual `remainder`.
    *   Designed for **stack-allocated persistence**; avoids heap-based state management, ensuring compatibility with the CENTAUR zero-cost memory model.
*   **`halting_step` (The Decision Engine):**
    *   **AVX-512 Vectorization Target:** Designed to consume quantized tensors (`MXUINT8`/`MXINT8`).
    *   **Logic:** Computes $\sigma(W_{halt} \cdot x_q + b_{halt})$. The output is compared against a threshold to toggle the `should_halt` boolean.
    *   **Zero-Cost Integration:** By operating on quantized inputs, it minimizes memory bandwidth pressure, allowing the halting decision to be computed in-register before the next layer's activation.

#### 3. AVX-512 & Physics Architecture Alignment
*   **Heapless Execution:** The `HaltingState` struct is POD (Plain Old Data), allowing it to reside in the stack frame or a pre-allocated scratchpad buffer. This eliminates dynamic memory allocation during the inference loop, preventing non-deterministic latency spikes.
*   **Branch Prediction Optimization:** The `should_halt` flag acts as a control signal for the instruction scheduler. In an AVX-512 context, this allows the engine to mask out inactive lanes or skip entire block computations, effectively implementing **hardware-level conditional execution**.
*   **Quantization-Aware:** By utilizing `MXINT8` weights, the halting gate maintains parity with the core compute pipeline, ensuring that the decision-making logic does not become a bottleneck compared to the primary matrix-multiplication kernels.

#### 4. Architectural Impact
This module is the primary gatekeeper for the CENTAUR engine's **"Compute-on-Demand"** philosophy. It transforms the network from a static graph into a state-machine, where the execution path is a function of the input data's information density.

---

## File: `nn\core\layers\sla.cpp`

### Architectural Analysis: Sparse Local Attention (SLA)

The `sla.cpp` module implements a high-throughput, cache-aware attention mechanism optimized for the CENTAUR Neural Engine. It prioritizes deterministic memory access patterns and SIMD saturation to minimize latency in transformer-based inference.

#### Core Purpose
The layer computes the weighted sum of value vectors based on query-key dot products. By enforcing a fixed `d_head=128` and `W=256` constraint, the implementation eliminates dynamic allocations, enabling a **heapless, stack-allocated execution model** that fits entirely within L2 cache, preventing costly DRAM round-trips.

#### Critical Functions & Optimization Strategy

*   **`sla_step_avx512`**: The primary execution path. It leverages **4-way ILP (Instruction Level Parallelism)** in Phase 1 to saturate the dual FMA ports of the AVX-512 unit. By unrolling the `d_head` loop, it maximizes register pressure to hide load-to-use latency.
*   **Phase 2 (Softmax)**: Implements a branchless, two-pass reduction. It utilizes `_mm512_reduce_max_ps` and `_mm512_reduce_add_ps` to collapse vectors, avoiding horizontal dependency stalls. The use of `nca::simd::avx512::exp_ps` suggests a custom, high-precision polynomial approximation (likely minimax) to maintain numerical stability without branching.
*   **Phase 3 (Weighted Accumulation)**: Employs a **register-tiling strategy**. By keeping 8 `__m512` accumulators in the register file, it transforms a memory-bound operation into a compute-bound one, effectively streaming `v_cache` through the FMA units.

#### Architectural Contributions
*   **Zero-Cost Abstraction**: The `CachePolicy` template provides compile-time prefetch distances (`PF`), allowing the compiler to inject `_mm_prefetch` hints based on the specific microarchitecture's L2/L3 latency profile without runtime overhead.
*   **Memory Alignment & Streaming**: The use of `__restrict` pointers and `_mm512_loadu_ps` (assuming cache-line aligned buffers) ensures the compiler generates optimal streaming store/load instructions, bypassing the cache hierarchy where appropriate to prevent pollution.
*   **Deterministic Execution**: By avoiding `std::vector` or heap-based buffers, the engine ensures constant-time execution, critical for real-time neural inference where jitter is unacceptable. The architecture treats the L2 cache as a software-managed scratchpad, aligning with the "physics-based" design philosophy of the CENTAUR engine.

---

## File: `nn\core\layers\sla.hpp`

### Architectural Analysis: `sla.hpp`

The `SLA` (Sparse Local Attention) layer implements a memory-efficient, sliding-window attention mechanism designed for high-throughput autoregressive inference. By constraining the attention span to a fixed $W=256$ window, it transforms the $O(N^2)$ complexity of standard self-attention into $O(N \cdot W)$, effectively eliminating the quadratic memory growth associated with long-context KV caches.

#### Core Architectural Pillars
*   **Zero-Cost Abstraction:** The `SLAConfig` struct is designed for compile-time or stack-allocated configuration, ensuring no heap allocation overhead during the inference loop.
*   **Cache-Friendly Layout:** The ring-buffer structure of `k_cache` and `v_cache` allows for contiguous memory access patterns, essential for saturating AVX-512 load/store units.
*   **Deterministic Memory Footprint:** By requiring the caller to provide the `scores` scratch buffer, the layer maintains a "heapless" profile, allowing the engine to pre-allocate workspace memory in a static arena.

#### Critical Function: `sla_step`
This function is the primary compute kernel, optimized for AVX-512 FMA (Fused Multiply-Add) pipelines.

1.  **Dot-Product Kernel:** Computes $q \cdot k_j$ using `_mm512_fmadd_ps`. With $d_{head}=128$, the loop processes 16 floats per iteration, requiring exactly 8 iterations per key vector.
2.  **Softmax Normalization:** Employs a two-pass approach:
    *   **Pass 1:** Find `max(scores)` using `_mm512_reduce_max_ps` to ensure numerical stability.
    *   **Pass 2:** Compute `exp(scores - max)` and accumulate the denominator.
3.  **Weighted Summation:** Performs the final projection $out = \sum (softmax_j \cdot v_j)$. This is the most compute-intensive phase, utilizing `_mm512_fmadd_ps` to accumulate into the output vector.

#### AVX-512 Optimization Strategy
*   **Vectorization:** The $d_{head}=128$ dimension is a multiple of the AVX-512 register width (16 floats), allowing for perfectly aligned, unrolled loops without tail-handling overhead.
*   **Data Alignment:** The `__restrict` qualifiers inform the compiler that pointers do not alias, enabling aggressive load-hoisting and instruction scheduling.
*   **Latency Hiding:** By structuring the kernel to operate on a fixed $W=256$, the compiler can effectively unroll the inner loops, allowing the CPU to hide load latency by interleaving independent FMA operations across the 32 available ZMM registers.

#### Integration into CENTAUR
This layer serves as the "physics" engine for the transformer decoder. By avoiding dynamic memory management, it ensures that the inference latency is strictly deterministic—a requirement for real-time neural execution. The design forces the KV cache to act as a circular buffer, minimizing cache misses and maximizing the utilization of the L1/L2 cache hierarchy.

---

## File: `nn\core\layers\ssm.cpp`

### Architectural Analysis: `nn/core/layers/ssm.cpp`

This module implements the core state-space model (SSM) update logic for the CENTAUR Neural Engine. It is designed for high-throughput, memory-bound inference where the working set (1.1MB) exceeds L1/L2 cache capacity, necessitating strict adherence to cache-line alignment and prefetch-aware scheduling.

#### Core Purpose
The layer performs the recurrent update $h_t = A \odot h_{t-1} + B \odot x_t$ and the projection $y_t = \sum(h_t \odot C)$. By maintaining $h$ in-place, the architecture minimizes memory traffic, transforming a standard SSM into a streaming kernel optimized for AVX-512 FMA throughput.

#### Critical Functions

*   **`ssm_step_avx512`**: The primary execution path. It utilizes a **Tile-based Processing** strategy (192-channel tiles) to maintain temporal locality.
    *   **ILP Optimization**: Employs 16-way unrolling to saturate the execution ports.
    *   **Reduction Strategy**: Replaces costly horizontal reductions with `_mm512_reduce_add_ps` only after accumulating the full state vector, effectively hiding the latency of the reduction tree behind the FMA pipeline.
*   **`ssm_step_scalar`**: Serves as the fallback/reference implementation. It is strictly structured to allow the compiler to perform auto-vectorization if AVX-512 is unavailable, maintaining the "zero-cost" abstraction principle.
*   **`mx_fused_ssm_silu_quantize` (Commented)**: Represents the future-state "Fused Kernel" architecture. It demonstrates the transition from standard FP32 compute to **Quantized Streaming**, where SiLU activation and E8M0 scaling are fused into the register-pressure-optimized loop to eliminate intermediate memory round-trips.

#### Architectural Contributions

1.  **Heapless Memory Management**: The architecture relies on caller-provided buffers (`__restrict` pointers), ensuring zero heap allocation during the inference step. This is critical for real-time deterministic latency.
2.  **Cache-Aware Prefetching**: By integrating `nca::simd::CachePolicy`, the kernel injects `_mm_prefetch` hints at the optimal distance (8 cache lines) to mask DDR4 latency, effectively turning a memory-bound operation into a compute-bound one.
3.  **Branchless Execution**: The logic avoids conditional branching within the inner loops. Even in the quantized path, clamping and scaling are implemented via `_mm512_min_epi32`/`max_epi32` to prevent pipeline stalls.
4.  **SIMD Dispatch**: The `ssm_step` entry point acts as a lightweight runtime dispatcher, ensuring that the most efficient instruction set (AVX-512) is selected without the overhead of virtual function tables or complex object-oriented patterns.

---

## File: `nn\core\layers\ssm.hpp`

### Architectural Analysis: `ssm.hpp`

This header defines the interface for the **Selective State-Space Model (SSM)** primitive within the CENTAUR engine. It serves as the temporal recurrence backbone, designed for high-throughput inference where state transitions must occur within the L1 cache boundary to avoid memory-wall latency.

#### Core Purpose
The SSM implementation facilitates linear recurrence relations ($h_t = Ah_{t-1} + Bx_t$) optimized for **AVX-512 register-level state persistence**. By maintaining the hidden state $h$ in ZMM registers rather than spilling to DRAM, the architecture achieves near-theoretical peak FLOPs for sequential processing.

#### Critical Functions

*   **`ssm_step`**: The primary compute kernel. It executes the state update and projection.
    *   **Architectural Strategy**: Designed for **loop unrolling and vectorization** over `d_inner`. The `__restrict` qualifiers are mandatory to allow the compiler to alias-analyze the pointers, enabling the use of `vmovaps` and `vfmadd213ps` instructions without load-store dependency stalls.
    *   **Memory Layout**: Expects contiguous memory blocks to facilitate **AVX-512 gather/scatter or aligned loads**, minimizing cache line splits.

*   **`mx_fused_ssm_silu_quantize_step` (Commented)**: Represents the **Horizontal Fusion** strategy.
    *   **Architectural Strategy**: Eliminates intermediate write-backs to the L1/L2 cache. By fusing the activation (SiLU) and quantization (FP32 $\rightarrow$ INT8) into the same execution pipeline as the SSM update, the engine keeps the intermediate result in the register file. This reduces the memory footprint by 4x and eliminates the bandwidth bottleneck of the quantization pass.

#### Contribution to Zero-Cost, Heapless Architecture

1.  **Deterministic Memory Footprint**: The `SSMConfig` struct enforces a static, stack-allocated or pre-allocated buffer approach. By avoiding dynamic allocation within the step, the engine eliminates non-deterministic latency spikes caused by heap fragmentation or allocator locks.
2.  **Register-Centric Execution**: The architecture treats the hidden state $h$ as a persistent register-resident entity. This "heapless" design ensures that the state transition is a pure arithmetic operation, mapping directly to the CPU's execution ports (Port 0/1/5) without triggering cache-coherency traffic.
3.  **Instruction-Level Parallelism (ILP)**: By exposing the `A`, `B`, and `C` matrices as raw pointers, the architecture allows the compiler to generate **fused multiply-add (FMA) chains**, maximizing the utilization of the AVX-512 unit's throughput while maintaining a minimal binary footprint.

---

## File: `nn\core\log.hpp`

### Architectural Analysis: `nca::log`

This module serves as the telemetry backbone for the CENTAUR Neural Engine. It prioritizes **compile-time metadata resolution** and **minimal instruction footprint** to ensure logging does not perturb the cache-locality or pipeline throughput of AVX-512 compute kernels.

#### Core Design Philosophy
*   **Zero-Cost Abstraction:** By leveraging `std::source_location`, the logger eliminates the need for preprocessor macros (`__FILE__`, `__LINE__`), allowing the compiler to treat log calls as standard function calls that are easily inlined or pruned.
*   **Heapless Execution:** The implementation avoids `std::string` or dynamic allocation, relying entirely on `std::string_view` and stack-allocated metadata. This prevents non-deterministic latency spikes during high-frequency neural inference.
*   **Branch Prediction Friendly:** The `g_min_level` check acts as a static filter. In optimized builds, the compiler can hoist this check or eliminate the call entirely if the level is constant-folded, ensuring zero overhead for disabled log levels.

#### Critical Components

*   **`emit()` (The Hot Path):**
    *   Acts as the central sink. The `std::source_location` parameter is defaulted at the call site, effectively embedding the metadata into the caller's stack frame.
    *   The path-stripping logic uses `std::string_view::find_last_of`, which the compiler optimizes into a tight loop or SIMD-accelerated search, minimizing the overhead of string manipulation.
*   **`level_tag()`:**
    *   Implemented as a `constexpr` lookup. It maps the `Level` enum to a static string literal, ensuring no runtime string construction occurs.
*   **Convenience Wrappers:**
    *   These provide a clean API surface while maintaining the `source_location` propagation chain. They are marked `inline` to ensure they disappear into the caller's instruction stream, maintaining the "zero-cost" requirement.

#### Integration with AVX-512 Physics Architecture
In the context of the CENTAUR engine, this logger is designed to be **non-intrusive to the execution pipeline**:
1.  **Instruction Cache Preservation:** By keeping the logging logic compact and inlined, we prevent "code bloat" that would otherwise evict critical AVX-512 kernel loops from the L1i cache.
2.  **Register Pressure:** The use of `std::string_view` (a simple pointer-length pair) ensures that passing log metadata does not spill registers, preserving the register file for heavy vector operations.
3.  **Deterministic Latency:** The absence of heap allocation and synchronization primitives (like `std::mutex`) ensures that logging—when enabled—does not introduce jitter into the real-time neural processing loop. 

**Architectural Recommendation:** To further optimize for high-throughput scenarios, consider adding a `consteval` check or a `[[likely]]` attribute to the `level < g_min_level` branch to ensure the compiler aggressively prunes the logging path from the hot-path instruction flow.

---

## File: `nn\core\normalization.cpp`

### Architectural Analysis: `nn/core/normalization.cpp`

This module implements Root Mean Square Normalization (RMSNorm), a critical layer in transformer-based architectures, optimized for the CENTAUR Neural Engine’s zero-cost, heapless execution model.

#### 1. Core Purpose
The module provides a high-performance, hardware-agnostic interface for tensor normalization. By abstracting SIMD dispatch, it ensures that the compute-heavy RMSNorm operation saturates the execution units of the underlying silicon (AVX-512/AVX2) while providing a scalar fallback for portability.

#### 2. Critical Functions

*   **`rmsnorm_scalar`**: Serves as the baseline implementation. It utilizes 8x loop unrolling to mitigate pipeline stalls and maximize instruction-level parallelism (ILP) on non-SIMD architectures. It minimizes memory latency by performing a single pass for the square sum and a second pass for the scaling operation, maintaining strict cache locality.
*   **`rmsnorm`**: Acts as the high-level entry point. It enforces memory safety via `std::span` and utilizes the `NCA_DISPATCH_KERNEL` macro to perform runtime CPU feature detection. This ensures the system selects the optimal kernel (AVX-512 vs. AVX2 vs. Scalar) without incurring virtual function overhead or heap allocations.

#### 3. Contribution to Zero-Cost/Heapless Architecture

*   **Zero-Cost Dispatch**: The `NCA_DISPATCH_KERNEL` macro facilitates compile-time/link-time selection or efficient runtime branching. By avoiding `std::function` or polymorphic objects, it eliminates pointer indirection and vtable lookups, keeping the instruction cache footprint minimal.
*   **Heapless Execution**: The implementation operates exclusively on raw pointers provided by `std::span`. No dynamic memory allocation occurs within the normalization pipeline, ensuring deterministic execution time and preventing fragmentation—a requirement for real-time neural inference.
*   **SIMD Alignment**: By leveraging `__restrict` qualifiers, the compiler is signaled that no pointer aliasing occurs, enabling aggressive vectorization and load-store optimization. The architecture assumes the caller manages memory alignment, allowing the kernels to utilize aligned AVX-512 load/store instructions (`vmovaps`) for maximum throughput.
*   **Instruction Pipeline Efficiency**: The use of `[[likely]]` and `[[unlikely]]` attributes provides the branch predictor with static hints, reducing pipeline flushes during tensor size validation and loop execution.

---

## File: `nn\core\normalization.hpp`

### Architectural Analysis: `nca::math::rmsnorm`

The `rmsnorm.hpp` header defines the interface for the **CENTAUR Neural Engine’s** normalization primitive. It serves as the abstraction layer between high-level tensor operations and hardware-specific SIMD kernels.

#### 1. Core Purpose
The function implements **Root Mean Square Normalization**, a critical component in Transformer-based architectures (e.g., Llama, Mistral) that stabilizes hidden state activations. By operating on `std::span`, the interface enforces memory safety and bounds checking without introducing heap allocations or pointer decay, maintaining the "zero-cost" requirement.

#### 2. Architectural Strategy
*   **Zero-Cost Abstraction:** The use of `std::span` allows the compiler to elide bounds checks in hot loops via loop unrolling and vectorization, provided the input sizes are known at compile-time or aligned correctly.
*   **Heapless Execution:** By accepting spans, the function operates on pre-allocated memory buffers (typically managed by a static arena or stack-allocated scratchpads), eliminating runtime `malloc` overhead during the inference pass.
*   **Dispatch Mechanism:** The implementation is designed for **Multi-Versioned Dispatch**. The backend likely utilizes `ifunc` or function pointers initialized at load-time to select the optimal instruction set:
    *   **AVX-512 (ZMM):** Utilizes `_mm512_reduce_add_ps` and `_mm512_rsqrt14_ps` for high-throughput, low-latency normalization.
    *   **AVX2 (YMM):** Fallback for legacy hardware, utilizing `_mm256_hadd_ps` and reciprocal approximations.
    *   **Scalar:** A strictly compliant C++20 fallback for portability.

#### 3. Critical Performance Considerations
*   **Numerical Stability:** The `eps` parameter is critical for preventing division-by-zero in the inverse square root calculation. The AVX-512 implementation should leverage `_mm512_rsqrt14_ps` followed by a single Newton-Raphson iteration to maintain high precision while maximizing throughput.
*   **Memory Alignment:** To achieve peak performance on AVX-512, the `std::span` buffers must be 64-byte aligned. The architecture assumes the caller provides aligned memory, allowing the kernel to use `_mm512_load_ps` (aligned) rather than `_mm512_loadu_ps` (unaligned), saving cycles on memory controller overhead.
*   **Cache Locality:** The function is designed to be cache-oblivious, processing the last dimension in a single pass to maximize L1/L2 cache hit rates, essential for the high-bandwidth requirements of the CENTAUR engine.

#### 4. Integration with CENTAUR
This header acts as the **Hardware Abstraction Layer (HAL)** entry point. By decoupling the interface from the implementation, the engine can swap the underlying math kernels (e.g., switching from `float` to `bfloat16` or `int8` quantization) without modifying the high-level neural network graph logic.

---

## File: `nn\core\simd\avx2_kernels.cpp`

### Architectural Analysis: `avx2_kernels.cpp`

This module serves as the high-throughput SIMD backend for the CENTAUR Neural Engine, implementing critical activation and normalization primitives. It bridges the gap between high-level tensor operations and hardware-level execution units, prioritizing cache locality and instruction-level parallelism (ILP).

#### Core Design Philosophy
*   **Zero-Cost Abstraction:** The implementation avoids heap allocations and dynamic dispatch, relying on static loop unrolling and `__restrict` pointers to enable aggressive compiler alias analysis and vectorization.
*   **Cache-Aware Execution:** Explicit `_mm_prefetch` hints mitigate memory latency by overlapping data movement with compute, essential for maintaining saturation on the execution ports.
*   **SIMD-to-Scalar Fallback:** The "likely/unlikely" attribute usage ensures the hot path (aligned/multiple-of-32 blocks) remains in the instruction cache, while scalar tail-processing handles arbitrary tensor dimensions without branching penalties.

#### Critical Function Breakdown

*   **`rmsnorm` (Root Mean Square Normalization):**
    *   **Mechanism:** Employs a two-pass approach. The first pass computes the sum of squares using four independent accumulators (`s0-s3`) to break dependency chains and maximize throughput on FMA ports.
    *   **Optimization:** Implements a Newton-Raphson refinement step on the hardware `_mm256_rsqrt_ps` result to achieve single-precision accuracy without the latency of a full `_mm256_div_ps` instruction.
    *   **Throughput:** The second pass fuses the inverse-square-root scaling and weight multiplication into a single FMA-heavy pipeline, minimizing register pressure.

*   **`silu` (Sigmoid Linear Unit):**
    *   **Mechanism:** Acts as a vectorized wrapper for the `silu_ps` intrinsic (likely defined in `avx2_math.hpp`).
    *   **Optimization:** By processing 32 floats per iteration (4x `__m256`), it maximizes the utilization of the AVX2 execution ports. The design assumes the underlying `silu_ps` uses a minimax polynomial approximation (e.g., Remez algorithm) to avoid expensive `exp()` calls, maintaining the "zero-cost" requirement.

#### Contribution to AVX-512 Physics Architecture
While currently targeting AVX2, this code provides the structural template for the upcoming AVX-512 migration:
1.  **Register Width Scaling:** The 32-float loop structure maps directly to `zmm` registers (16 floats per `zmm`), allowing for a 2x throughput increase upon porting to AVX-512.
2.  **Masking Potential:** The current scalar tail-processing logic is the primary candidate for replacement by AVX-512 **opmask registers (`k0-k7`)**, which will eliminate the need for the `rem` loop entirely, further reducing code size and branch mispredictions.
3.  **Memory Alignment:** The use of `_mm256_loadu_ps` provides the necessary flexibility for non-aligned tensor buffers, a prerequisite for the heapless, zero-copy memory management required by the CENTAUR engine.

---

## File: `nn\core\simd\avx2_kernels.hpp`

### Architectural Analysis: `avx2_kernels.hpp`

This header defines the SIMD-accelerated primitive layer for the CENTAUR Neural Engine. Despite the `avx2` namespace, it serves as the hardware-abstraction interface for the engine's vector execution unit, designed to be dispatched by the higher-level AVX-512 orchestration layer.

#### Core Purpose
The module provides **data-parallel kernels** optimized for cache-locality and register-pressure management. By utilizing `__restrict` pointers, it guarantees the compiler that no pointer aliasing occurs, enabling aggressive loop unrolling and instruction scheduling without the overhead of heap-allocated buffers or dynamic dispatch.

#### Critical Functions

*   **`rmsnorm`**: Implements Root Mean Square Layer Normalization.
    *   **Architectural Role**: Reduces internal covariate shift.
    *   **Optimization Strategy**: Leverages fused multiply-add (FMA) chains to compute the sum of squares in a single pass. By maintaining the inverse square root in a vector register, it avoids costly division operations, ensuring the normalization factor is applied with minimal latency.
*   **`silu`**: Implements the Sigmoid-Weighted Linear Unit activation function ($x \cdot \sigma(x)$).
    *   **Architectural Role**: Provides non-linear gating for transformer blocks.
    *   **Optimization Strategy**: Minimizes transcendental instruction latency. It utilizes a polynomial approximation for the sigmoid component, ensuring high throughput within the execution pipeline while maintaining numerical stability for FP32 precision.

#### Zero-Cost, Heapless Integration
*   **Memory Model**: The interface operates strictly on raw memory buffers (`float*`), allowing the engine to perform **in-place transformations** on pre-allocated scratchpad memory. This eliminates heap fragmentation and garbage collection overhead.
*   **Zero-Cost Abstraction**: By leveraging `std::span` (in the interface) and raw pointers (in the implementation), the engine avoids the metadata overhead of `std::vector`. The compiler can inline these kernels directly into the compute graph, effectively treating the SIMD operations as intrinsic extensions of the host code.
*   **Alignment & Vectorization**: The architecture assumes data is aligned to 64-byte boundaries (AVX-512 cache line alignment), allowing the compiler to emit `vmovaps` (aligned move) instructions, bypassing the performance penalty of unaligned memory access.

---

## File: `nn\core\simd\avx2_math.hpp`

### Architectural Analysis: `nca::simd::avx2`

This module provides the low-latency, branchless math primitives required for the CENTAUR Neural Engine’s inference pipeline. By bypassing standard library calls, it ensures deterministic execution timing and avoids heap allocation or stack-frame overhead, maintaining the "zero-cost" abstraction requirement.

#### 1. `exp_ps` (Minimax Exponential)
*   **Mechanism:** Implements a range-reduced minimax polynomial approximation (degree 6) combined with IEEE-754 bit-level exponent scaling.
*   **Optimization:** Uses `_mm256_round_ps` for range reduction and `_mm256_slli_epi32` to inject the integer exponent directly into the floating-point representation. This avoids expensive `powf` calls and maintains high throughput by keeping data in registers.
*   **Constraint:** Clamped to $[-88.37, 88.37]$ to prevent underflow/overflow, ensuring stability in deep neural network activations.

#### 2. `silu_ps` (Sigmoid Linear Unit)
*   **Mechanism:** Computes $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$.
*   **Optimization:** 
    *   **Reciprocal Refinement:** Employs `_mm256_rcp_ps` followed by a single Newton-Raphson iteration (`y1 = y0 * (2 - den * y0)`) to achieve near-full precision division without the latency of the `divps` instruction.
    *   **FMA Utilization:** Leverages `_mm256_fnmadd_ps` to perform the fused multiply-negate-add in the Newton-Raphson step, minimizing port pressure on the execution units.

#### 3. Integration with AVX-512 Architecture
*   **Zero-Cost Strategy:** These functions operate strictly on `__m256` registers, facilitating seamless inlining into the broader CENTAUR compute graph.
*   **Transition Path:** While currently AVX2-bound, the logic is architecturally aligned for a "promotion" to AVX-512 (`zmm` registers). The transition requires only replacing `_mm256` intrinsics with `_mm512` equivalents and utilizing `_mm512_rcp14_ps` for higher-precision reciprocal approximation, which is natively supported in the AVX-512 instruction set.
*   **Heapless Guarantee:** By design, these functions are stack-allocated and register-resident, ensuring they satisfy the strict memory-safety and performance constraints of the CENTAUR engine's real-time physics simulation environment.

---

## File: `nn\core\simd\avx512_kernels.cpp`

### Architectural Analysis: `avx512_kernels.cpp`

This module serves as the high-throughput compute backbone for the CENTAUR Neural Engine, prioritizing instruction-level parallelism (ILP) and cache-line utilization to minimize latency in transformer-based inference.

#### Core Design Philosophy
*   **Zero-Cost Abstraction:** The implementation avoids heap allocations and dynamic dispatch, relying on static loop unrolling and compiler-hinted branch prediction (`[[likely]]`/`[[unlikely]]`) to ensure the instruction pipeline remains saturated.
*   **Memory-Bound Optimization:** By utilizing `_mm_prefetch` with `_MM_HINT_T0`, the kernel proactively pulls data into L1 cache, effectively hiding memory latency for large tensor operations.
*   **Precision-Latency Trade-off:** The use of `_mm512_rsqrt14_ps` followed by a single Newton-Raphson iteration provides sufficient precision for neural network activations while bypassing the high-latency `vsqrtps` and `vdivps` instructions.

#### Critical Function Breakdown

**`rmsnorm` (Root Mean Square Normalization)**
*   **Two-Pass Strategy:** Decouples the reduction (sum of squares) from the normalization (scaling). This maximizes throughput by allowing the CPU to execute the reduction independently of the weight-application phase.
*   **Register Tiling:** Processes 64 floats per iteration (4x 512-bit registers), maximizing the utilization of the AVX-512 register file and reducing loop overhead.
*   **Masked Tail Handling:** Employs `__mmask16` for boundary conditions, eliminating the need for scalar cleanup loops and maintaining SIMD execution flow even for non-aligned tensor sizes.

**`silu` (Sigmoid Linear Unit)**
*   **Activation Throughput:** Leverages the `silu_ps` intrinsic (assumed to be a vectorized implementation of $x \cdot \sigma(x)$).
*   **In-Place Transformation:** Operates directly on the input buffer, minimizing cache pressure and memory bandwidth consumption.
*   **Instruction Scheduling:** The 4x unrolling pattern ensures that the execution units (FMA/ALU) are kept busy, effectively hiding the latency of the transcendental operations required for the sigmoid component.

#### Architectural Impact
This implementation facilitates a **heapless execution model** by assuming caller-managed buffers, which is critical for real-time inference where non-deterministic latency from the memory allocator is unacceptable. By aligning the data access patterns with the 64-byte cache line size of modern x86 architectures, the kernel achieves near-peak theoretical throughput for memory-bound operations.

---

## File: `nn\core\simd\avx512_kernels.hpp`

### Architectural Analysis: `avx512_kernels.hpp`

This header defines the SIMD-accelerated primitive layer for the CENTAUR Neural Engine. It enforces a strict **data-oriented design** by decoupling compute kernels from memory management, ensuring the engine remains heapless and cache-coherent.

#### Core Architectural Principles
*   **Pointer Aliasing Control:** The use of `__restrict` is foundational. It informs the compiler that `out`, `in`, and `weight` buffers are disjoint, enabling the backend to bypass load-store dependency checks and maximize instruction-level parallelism (ILP) via aggressive loop unrolling and software pipelining.
*   **Zero-Cost Abstraction:** By operating on raw pointers and `size_t` bounds, the interface avoids the overhead of object-oriented wrappers or dynamic dispatch, allowing the compiler to inline these kernels directly into the neural graph execution path.
*   **Memory Alignment Assumptions:** The interface implies that callers provide memory aligned to 64-byte boundaries, allowing the use of `_mm512_load_ps` and `_mm512_store_ps` without penalty-heavy unaligned access instructions.

#### Functional Breakdown

*   **`rmsnorm` (Root Mean Square Normalization):**
    *   **Purpose:** Stabilizes hidden state activations by scaling inputs by their inverse root mean square.
    *   **SIMD Strategy:** Utilizes horizontal reduction instructions (`_mm512_reduce_add_ps`) to compute the variance across the vector, followed by a reciprocal square root approximation (`_mm512_rsqrt14_ps`) refined via Newton-Raphson iterations to maintain precision while minimizing latency.
*   **`silu` (Sigmoid Linear Unit):**
    *   **Purpose:** Provides non-linear activation, critical for gating mechanisms in Transformer architectures.
    *   **SIMD Strategy:** Implemented as $x \cdot \sigma(x)$. The kernel leverages the AVX-512 exponential approximation instructions to compute the sigmoid component, effectively mapping the activation function across 16 single-precision floats per cycle, maximizing throughput in the compute-bound phase of the forward pass.

#### Integration with CENTAUR Physics Architecture
This module acts as the **Compute Backend**. By keeping these kernels pure and stateless, the engine maintains a deterministic execution profile. The absence of heap allocations within these functions ensures that the neural engine can operate within pre-allocated memory arenas, satisfying the "zero-cost" requirement for real-time inference environments where latency jitter is unacceptable.

---

## File: `nn\core\simd\avx512_math.hpp`

### Architectural Analysis: `avx512_math.hpp`

This header serves as the low-latency mathematical foundation for the CENTAUR Neural Engine, prioritizing instruction-level parallelism (ILP) and pipeline throughput over high-precision transcendental accuracy.

#### Core Design Philosophy
*   **Zero-Cost Abstraction:** By utilizing `inline` primitives, the compiler integrates these operations directly into the caller's register allocation graph, eliminating stack frame overhead and enabling cross-function instruction scheduling.
*   **Heapless Execution:** All operations are strictly register-bound (`__m512`), ensuring deterministic execution time and cache-locality, critical for real-time neural inference.
*   **Branchless Pipeline:** By avoiding conditional jumps, the implementation prevents pipeline stalls and branch misprediction penalties, maintaining a steady throughput of 16 `float` operations per cycle.

#### Critical Primitives

**`exp_ps` (Minimax Approximation)**
*   **Mechanism:** Employs a range-reduction strategy using `_mm512_roundscale_ps` to isolate the fractional component ($z$) and the integer exponent ($e$).
*   **Optimization:** Uses a 6th-degree minimax polynomial for the fractional part, evaluated via FMA (Fused Multiply-Add) chains. This minimizes rounding error while maximizing throughput by leveraging the 2-cycle latency of AVX-512 FMA units.
*   **Scaling:** The final `_mm512_scalef_ps` instruction performs a hardware-accelerated power-of-two scaling, effectively reconstructing the exponential result without costly bit-manipulation or table lookups.

**`silu_ps` (Sigmoid Linear Unit)**
*   **Mechanism:** Implements $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$.
*   **Optimization:** Instead of a standard division, it utilizes `_mm512_rcp14_ps` (Reciprocal Approximation) followed by a single Newton-Raphson iteration (`_mm512_fnmadd_ps`). This provides 14-bit precision—sufficient for neural activation functions—at a fraction of the latency of the `vdivps` instruction.
*   **Efficiency:** By combining the reciprocal approximation with the FMA-based polynomial `exp_ps`, the function maintains a high throughput-to-latency ratio, essential for the bottlenecked activation layers of the CENTAUR engine.

#### Architectural Impact
This module enables the engine to bypass standard library calls (which are often scalar or non-inlinable) and maintain a "pure-SIMD" execution path. By keeping the math within the AVX-512 register file, the architecture avoids the "SIMD-to-Scalar" transition penalty, ensuring that the neural engine remains compute-bound rather than memory- or latency-bound.

---

## File: `nn\core\simd\cache_policy.hpp`

### Architectural Analysis: `cache_policy.hpp`

This header serves as the **Memory Orchestration Layer** for the CENTAUR Neural Engine. It abstracts the hardware-software interface, bridging the gap between static kernel compilation and dynamic hardware topology.

#### 1. Core Purpose: Deterministic Memory Orchestration
The engine enforces a strict **zero-allocation, heapless policy**. By calculating memory footprints at compile-time (`CachePolicy`) or via lightweight stack-allocated structures (`RuntimeCachePolicy`), it eliminates runtime dynamic memory management. This ensures that the AVX-512 execution units are never stalled by page faults or allocator overhead.

#### 2. Critical Components

*   **`CacheStrategy` (State Machine):**
    *   **`L1_HOT`**: Maximizes temporal locality. Disables prefetching to prevent cache pollution, allowing the pipeline to saturate the L1d bandwidth.
    *   **`L2_STREAM`**: Optimizes for L2-resident working sets. Uses software prefetching to hide latency of L1 misses while maintaining cache coherence.
    *   **`DDR4_NT`**: Implements Non-Temporal (NT) store semantics. Bypasses the cache hierarchy for write-heavy operations, preventing "cache thrashing" when the working set exceeds L2 capacity.

*   **`tail_mask` (Branchless Predication):**
    *   Utilizes `_bzhi_u32` (or equivalent bit-shifting) to generate AVX-512 opmask registers.
    *   **Architectural Impact**: Eliminates the branch misprediction penalty associated with tail-processing in SIMD loops. By converting `remaining` elements into a bitmask, the kernel executes a single, uniform instruction stream regardless of input size.

*   **`working_set_bytes` (Static Footprint Analysis):**
    *   A template-metaprogramming utility that calculates the exact memory footprint of a kernel at compile-time.
    *   **Architectural Impact**: Enables the compiler to perform aggressive loop unrolling and register allocation based on known cache constraints, effectively "tuning" the binary to the specific cache hierarchy of the target CPU.

#### 3. Integration with AVX-512 Physics
The architecture relies on this header to define the **Data Movement Contract**:

1.  **Zero-Cost Abstraction**: The `CachePolicy<N>` template resolves to constant values at compile-time. The compiler treats these as immediate values, resulting in zero runtime overhead for policy selection.
2.  **Cache-Aware Tiling**: By providing `tile_lines` and `prefetch_dist`, the engine allows kernels to partition large tensors into cache-friendly blocks. This ensures that the AVX-512 FMA (Fused Multiply-Add) units are fed at the maximum possible throughput, minimizing the "Memory Wall" effect.
3.  **Hardware-Agnosticism**: The `RuntimeCachePolicy` allows the engine to adapt to varying L1/L2 sizes across different CPU microarchitectures (e.g., Golden Cove vs. Gracemont) without recompilation, maintaining peak performance across heterogeneous deployments.

#### 4. Summary of Design Philosophy
This module enforces **Data-Oriented Design (DOD)**. By treating cache hierarchy as a first-class citizen in the type system, it forces developers to define the memory footprint of a kernel before execution. This results in a predictable, high-throughput execution environment where the CPU's cache controller is treated as a managed resource rather than a black box.

---

## File: `nn\core\simd\dispatch.cpp`

### Architectural Analysis: `nca::simd::dispatch`

This module serves as the **Hardware Abstraction Layer (HAL)** for the CENTAUR Neural Engine. It provides a static, zero-overhead mechanism to resolve the optimal SIMD instruction set architecture (ISA) at runtime, ensuring the engine maximizes throughput on modern x86_64 silicon without incurring heap allocations or dynamic dispatch penalties.

#### Core Purpose
*   **ISA Negotiation:** Bridges the gap between raw CPU capabilities and the engine’s compute kernels.
*   **State Persistence:** Uses a `static const` singleton pattern to ensure detection logic executes exactly once, providing O(1) access to hardware capabilities for the remainder of the application lifecycle.
*   **Override Injection:** Enables deterministic testing and debugging by allowing the injection of a specific `Backend` via `g_override_backend`, bypassing hardware detection.

#### Critical Function Breakdown

*   **`cpuid` / `os_supports_avx512`**: These functions perform low-level register interrogation. By checking `XCR0` (Extended Control Register), the code verifies that the OS kernel has explicitly enabled the saving/restoring of ZMM registers, preventing illegal instruction faults during context switches.
*   **`detect_impl`**: A bit-masking engine that parses `CPUID` leaf 7. It specifically isolates `AVX512F` (Foundation) and `AVX512_VNNI` (Vector Neural Network Instructions). The latter is critical for the CENTAUR engine, as it enables 8-bit integer dot-product acceleration (VPDPBUSD), which is the primary driver for inference performance.
*   **`best_backend`**: The decision-making heuristic. It prioritizes VNNI-capable AVX-512 for high-density compute, falls back to AVX2 for legacy compatibility, and defaults to Scalar for portability.

#### Contribution to "Zero-Cost, Heapless" Architecture
*   **Static Dispatch:** By resolving the `Backend` enum at the entry point of a kernel call, the engine can use `if constexpr` or template specialization in the calling code to eliminate branch mispredictions and function pointer overhead.
*   **Memory Efficiency:** The implementation avoids `std::string` or heap-allocated objects, relying on stack-based `CPUInfo` structs and `constexpr` string literals. This ensures the dispatch logic is cache-friendly and suitable for embedded or high-frequency trading environments where jitter is unacceptable.
*   **Binary Footprint:** The use of `std::optional` for the override mechanism is localized to the dispatch module, ensuring the hot path remains a simple integer comparison against the `static const` detection result.

---

## File: `nn\core\simd\dispatch.hpp`

### Architectural Analysis: `nca::simd::Dispatcher`

The `dispatch.hpp` module implements a **static-to-dynamic bridge** for the CENTAUR Neural Engine, enabling runtime ISA selection without sacrificing the performance of static dispatch.

#### 1. Core Purpose
The system provides a **thread-safe, lazy-initialized function pointer cache**. It bridges the gap between compile-time kernel specialization (Scalar, AVX2, AVX-512) and runtime hardware capability, ensuring that the overhead of ISA detection is amortized to a single atomic load after the first invocation.

#### 2. Critical Components
*   **`Dispatcher<FuncPtr>`**: A template-based functor that encapsulates the kernel selection logic. By using `std::atomic<FuncPtr>` with `memory_order_relaxed`, it avoids expensive cache-coherency traffic (fences) on subsequent calls, effectively reducing the dispatch cost to a single pointer dereference.
*   **`NCA_DISPATCH_KERNEL` Macro**: Implements the **"Static-Local Singleton" pattern**. By declaring the `Dispatcher` as `static const` within the macro, the engine ensures the dispatcher is initialized exactly once per call site, maintaining a heapless memory footprint and avoiding global constructor order issues.
*   **`select_uncached()`**: The fallback mechanism for testing and override scenarios. It bypasses the cache to allow for dynamic ISA switching (e.g., forcing AVX2 on an AVX-512 machine for AMI validation).

#### 3. Zero-Cost/Heapless Physics Integration
*   **Zero-Cost Abstraction**: The `[[likely]]` and `[[unlikely]]` attributes guide the branch predictor to favor the cached path. In steady-state execution, the CPU executes a direct jump to the optimized kernel, effectively matching the performance of a static function call.
*   **Heapless Design**: The architecture relies entirely on stack-allocated or static-data-segment memory. There are no `std::function` objects or heap-allocated closures, preventing fragmentation and ensuring deterministic latency—a requirement for high-frequency neural inference.
*   **Instruction Set Agnosticism**: By abstracting the ISA behind a uniform interface, the physics engine can maintain a single high-level codebase while the `Dispatcher` handles the low-level SIMD vectorization requirements (e.g., 512-bit register utilization vs. 256-bit fallback).

#### 4. Architectural Critique
*   **Strengths**: The use of `std::memory_order_relaxed` is optimal here; since the function pointer is idempotent and the state is effectively immutable after the first write, we avoid the performance penalty of `seq_cst` atomics.
*   **Risk**: The `Dispatcher` assumes the function signature is identical across all backends. If the AVX-512 kernel requires different alignment or memory layout than the Scalar version, the abstraction may leak implementation details. Ensure that all kernels are wrapped in a common ABI-compatible interface.

---

## File: `nn\core\simd\memory.hpp`

### Architectural Analysis: `nca::simd::memory`

This module provides the foundational memory management layer for the CENTAUR Neural Engine, enforcing strict 64-byte alignment required for AVX-512 load/store operations (e.g., `vmovaps` on ZMM registers). It bridges the gap between standard C++ RAII and platform-specific aligned allocation.

#### Core Purpose
To eliminate cache-line split penalties and alignment faults in SIMD kernels by providing a RAII-compliant wrapper for `_aligned_malloc`. It ensures that heap-allocated neural weights and activation buffers are cache-line aligned, enabling optimal throughput for 512-bit wide vector instructions.

#### Critical Components

*   **`AlignedDeleter<T>` / `AlignedDeleter<T[]>`**:
    *   Implements a custom functor for `std::unique_ptr` to handle the non-standard deallocation path required by `_aligned_malloc`.
    *   **Destructor Orchestration**: Uses `if constexpr` to perform compile-time pruning of destructor calls for trivially destructible types, ensuring zero-overhead cleanup for POD-based neural tensors.
    *   **Array Safety**: The `T[]` specialization tracks element counts to ensure correct destruction of object arrays, preventing memory leaks in complex layer state objects.

*   **`make_aligned_unique<T>`**:
    *   **Factory Pattern**: Encapsulates the `malloc` -> `placement new` -> `unique_ptr` lifecycle.
    *   **Exception Safety**: The array-based factory implements a strong exception guarantee; if a constructor throws during initialization, it performs a partial rollback of previously constructed elements before freeing the buffer.
    *   **Type Erasure/Deduction**: Uses `std::enable_if_t` and `std::remove_extent_t` to provide a unified interface for both scalar and array allocations, maintaining strict type safety while abstracting the underlying alignment requirements.

#### Contribution to CENTAUR Architecture
*   **Zero-Cost Abstraction**: By leveraging `if constexpr` and template metaprogramming, the overhead of the deleter is effectively zero for POD types (e.g., `float32` buffers), matching the performance of raw pointers while providing full RAII safety.
*   **SIMD Alignment Guarantee**: By hardcoding the 64-byte alignment, it guarantees that every buffer allocated through this system is compatible with AVX-512's `vmovaps` (aligned move) instructions, preventing the performance degradation associated with `vmovups` (unaligned move) on older microarchitectures and ensuring optimal cache-line utilization.
*   **Heapless Philosophy**: While it uses the heap, it promotes "heapless" design patterns by encouraging the creation of long-lived, aligned buffers at the start of the inference pipeline, minimizing runtime allocations and fragmentation during the hot path of neural execution.

---

## File: `nn\core\spectral\fwht.cpp`

### Architectural Analysis: `nn/core/spectral/fwht.cpp`

This module implements an in-place, cache-oblivious Fast Walsh-Hadamard Transform (FWHT) optimized for the CENTAUR Neural Engine. It leverages AVX-512 to minimize instruction latency and maximize throughput in spectral domain operations.

#### Core Purpose
The FWHT is the spectral backbone of the CENTAUR engine, providing a computationally efficient alternative to the FFT for signal processing and feature extraction. By operating in-place, it maintains a **zero-cost memory footprint**, avoiding heap allocations and minimizing cache pressure—critical for high-frequency neural inference.

#### Critical Functions

*   **`fwht_inplace`**: The primary engine. It utilizes a hierarchical butterfly decomposition.
    *   **ILP Optimization**: The 2x unroll (32-float block) saturates the dual FMA ports of the AVX-512 pipeline, effectively hiding load-to-use latency.
    *   **Branching Strategy**: Uses a tiered approach (AVX-512 -> AVX2 -> Scalar) to ensure optimal vector width utilization based on the current transform stage (`len`).
*   **`ifwht_inplace`**: Implements the inverse transform by leveraging the symmetry of the Hadamard matrix. It performs a single-pass normalization using `_mm512_mul_ps`, ensuring the scaling operation is bound by memory bandwidth rather than compute latency.
*   **`butterfly_v16` / `butterfly_v8`**: Low-level vector primitives. By using `__restrict` pointers, these functions signal to the compiler that no pointer aliasing occurs, enabling aggressive load/store reordering and register renaming.

#### Architectural Contributions
*   **Zero-Cost/Heapless**: The implementation operates strictly on the provided `std::span`, ensuring the engine remains deterministic and stack-friendly.
*   **AVX-512 Pipeline Saturation**: By aligning the inner loops to 512-bit boundaries, the code maximizes the utilization of the ZMM register file, reducing the total cycle count per butterfly operation.
*   **Cache Locality**: The iterative structure respects the spatial locality of the input buffer, minimizing cache misses during the recursive-like stages of the transform.
*   **Hardware-Agnostic Dispatch**: The use of preprocessor guards (`__AVX512F__`) ensures the code remains portable across the CENTAUR hardware stack while defaulting to high-performance AVX2 paths on legacy silicon.

---

## File: `nn\core\spectral\fwht.hpp`

### Architectural Analysis: `fwht.hpp`

The `fwht.hpp` module provides the spectral backbone for the CENTAUR Neural Engine, enabling efficient signal decomposition and feature extraction within the frequency domain. By leveraging the Walsh-Hadamard Transform, the engine avoids the complex-number overhead of FFTs, operating exclusively on real-valued floating-point data.

#### Core Purpose
This module facilitates **Hadamard-domain processing**, essential for sparse neural representations and high-speed signal correlation. It serves as a mathematical primitive for operations requiring orthogonal basis projections without the computational cost of trigonometric lookups or transcendental function calls.

#### Critical Functions
*   **`fwht_inplace`**: Implements the Cooley-Tukey-style butterfly network for the Hadamard transform. Designed for cache-locality, it performs in-place bit-reversal or iterative butterfly stages to minimize memory traffic.
*   **`ifwht_inplace`**: Applies the inverse transform. Exploits the symmetry of the Hadamard matrix ($H_n = H_n^{-1}$), requiring only a final normalization pass ($1/N$) to restore signal amplitude.
*   **`ifwht_no_scale`**: Provides a raw inverse projection. Used in multi-stage neural pipelines where normalization is deferred to a subsequent layer to maintain precision and avoid redundant floating-point divisions.

#### AVX-512 & Zero-Cost Integration
*   **SIMD Vectorization**: The implementation is architected for `zmm` register utilization. By processing 16 `float` lanes per instruction, the butterfly stages map directly to `_mm512_add_ps` and `_mm512_sub_ps` operations, maximizing throughput on the CENTAUR execution units.
*   **Heapless Execution**: The `std::span` interface enforces memory safety without dynamic allocation. By operating on pre-allocated buffers (typically stack-allocated or arena-managed), the module ensures deterministic latency and zero-fragmentation, critical for real-time physics-informed neural inference.
*   **Cache-Aligned Throughput**: The iterative structure is optimized for L1/L2 cache line alignment, ensuring that the $O(N \log N)$ complexity is bounded by arithmetic throughput rather than memory stalls.

---

## File: `nn\core\spectral\kronecker_rls.cpp`

### Architectural Analysis: Kronecker-Factored RLS (KFW)

The `KroneckerRLSState` implements a memory-efficient Recursive Least Squares (RLS) variant. By approximating the $d \times d$ covariance matrix as a Kronecker product of two smaller matrices ($A \in \mathbb{R}^{dim_a \times dim_a}$ and $B \in \mathbb{R}^{dim_b \times dim_b}$), it reduces space complexity from $O(d^2)$ to $O(d)$, enabling high-dimensional spectral adaptation within the constrained memory footprint of the CENTAUR Neural Engine.

#### Core Architectural Components

*   **Kronecker Decomposition:** The state maintains $W_a, W_b$ (weights) and $A, B$ (inverse covariance factors). This factorization allows the engine to perform matrix-vector products as a sequence of smaller, cache-friendly operations, effectively bypassing the $O(d^2)$ bottleneck of standard RLS.
*   **AVX-512/256 Vectorization Strategy:** The implementation utilizes `_mm256_fmadd_ps` (FMA) for high-throughput arithmetic. While the current code uses AVX2 intrinsics, the memory layout is designed for 64-byte alignment, facilitating a trivial migration to AVX-512 `zmm` registers to double throughput per cycle.
*   **Stability & Numerical Integrity:**
    *   **Diagonal Regularization:** Injects $10^{-6}$ into the diagonal of $A$ and $B$ to prevent singularity during inversion updates.
    *   **Explosion Prevention:** Implements a hard-cap on covariance magnitude ($10^4$) to mitigate the "wind-up" effect common in recursive spectral estimators.
    *   **Bit-Level Sanitization:** Uses `std::bit_cast` to detect and mask `NaN`/`Inf` values, resetting corrupted weights to identity to ensure the engine remains "self-healing" without branching overhead.

#### Critical Function Breakdown

*   **`apply()`**: Executes the forward pass. It treats the input as a flattened matrix, applying $W_b$ via broadcast-multiplication and $W_a$ via contiguous dot products. This minimizes cache misses by maintaining spatial locality across the $dim_a$ dimension.
*   **`update()`**: Performs the recursive adaptation.
    *   **Gain Computation:** Calculates the Kalman gain $K$ using the Kronecker structure.
    *   **Weight Adaptation:** Updates $W_a$ and $W_b$ using the error vector $e$. The use of `alignas(64)` for local buffers ensures that the stack-allocated workspace remains aligned for optimal SIMD load/store operations.
    *   **Woodbury Decay:** Applies the forgetting factor $\lambda$ to the covariance factors, effectively performing a rank-1 update to the inverse covariance without explicit matrix inversion.

#### Integration with CENTAUR Physics Architecture

*   **Heapless Execution:** By using `nca::simd::make_aligned_unique` and stack-allocated buffers for intermediate calculations, the engine avoids non-deterministic heap allocations during the inference/training loop.
*   **Zero-Cost Abstraction:** The logic relies on compile-time dimension calculation and template-like behavior (via `dim_a`, `dim_b` initialization), ensuring that the compiler can unroll loops and inline the SIMD kernels directly into the neural pipeline.
*   **Deterministic Latency:** The absence of dynamic memory management and the use of fixed-size buffers ensure that the execution time is strictly proportional to $O(d)$, critical for real-time spectral processing in the CENTAUR engine.

---

## File: `nn\core\spectral\kronecker_rls.hpp`

### Architectural Analysis: `kronecker_rls.hpp`

The `KroneckerRLSState` implements a memory-efficient approximation of the Recursive Least Squares (RLS) algorithm by decomposing a large $d \times d$ covariance matrix into two smaller Kronecker factors ($A \in \mathbb{R}^{dim_a \times dim_a}$ and $B \in \mathbb{R}^{dim_b \times dim_b}$). This reduces the parameter space from $O(d^2)$ to $O(dim_a^2 + dim_b^2)$, enabling real-time spectral adaptation within the CENTAUR Neural Engine.

#### Core Architectural Components

*   **Tensor Decomposition:** By enforcing $W = W_a \otimes W_b$, the engine avoids the $O(d^3)$ inversion complexity of standard RLS. The state maintains inverse covariance factors $A$ and $B$, allowing for rank-1 updates that are computationally tractable for high-frequency spectral tracking.
*   **Memory Alignment:** The use of `nca::simd::aligned_unique_ptr` ensures that all weight and covariance buffers are aligned to 64-byte boundaries, a prerequisite for optimal AVX-512 `vmovaps` and `vmovups` throughput.
*   **Zero-Cost Integration:** The structure is designed for stack-friendly initialization or placement-new within pre-allocated memory pools, minimizing heap fragmentation during the engine's hot-path execution.

#### Critical Function Breakdown

*   **`update(const float* x, const float* target, float lambda, float eta)`**
    *   **Mechanism:** Implements the Kronecker-factored update rule. It performs a dual-factor rank-1 update on $A$ and $B$ using the forgetting factor $\lambda$ and learning rate $\eta$.
    *   **AVX-512 Optimization:** Designed to leverage `vfmadd213ps` (FMA) for the covariance update, effectively hiding latency by processing 16-wide float vectors across the Kronecker factors.

*   **`apply(const float* x, float* out)`**
    *   **Mechanism:** Executes the Kronecker product operator $y = \text{vec}(W_b X W_a^T)$.
    *   **AVX-512 Optimization:** Maps the matrix-matrix multiplication to a series of fused-multiply-add operations. By treating the input as a reshaped matrix, it maximizes cache locality and utilizes the 32 ZMM registers to keep intermediate products in-flight, bypassing the need for temporary buffers.

*   **`reset()`**
    *   **Mechanism:** Initializes the inverse covariance factors to identity matrices scaled by a regularization constant.
    *   **AVX-512 Optimization:** Uses `vbroadcastss` to fill the diagonal elements, ensuring the state is ready for immediate convergence without requiring iterative warm-up cycles.

#### Contribution to CENTAUR Physics Architecture
This module serves as the spectral "memory" of the engine. By offloading the heavy lifting of covariance tracking to the Kronecker domain, it maintains a constant memory footprint. This allows the engine to perform adaptive spectral filtering at the sample rate, maintaining the "zero-cost" requirement by ensuring that the computational cost per update is strictly deterministic and independent of the total model dimensionality.

---

## File: `nn\core\spectral\spectral_logic.cpp`

### Architectural Analysis: `spectral_logic.cpp`

This module implements the **Spectral Domain Update** layer for the CENTAUR Neural Engine. It shifts state representations from the spatial domain to the Walsh-Hadamard domain to perform low-rank recursive least squares (RLS) updates, effectively treating the neural state as a signal processing stream.

#### 1. Core Purpose
The module facilitates **Evidence Conserving Updates**. By operating in the spectral domain, it performs global state modulation via Kronecker-factored RLS, allowing the model to learn associative mappings without explicit weight matrices. This minimizes memory footprint by replacing dense layers with spectral operators.

#### 2. Critical Functions
*   **`spectral_logic_step`**: The primary orchestration routine. It manages the lifecycle of the spectral transformation pipeline.
    *   **Transformation Pipeline**: Orchestrates `fwht_inplace` (Fast Walsh-Hadamard Transform) to map inputs into the spectral domain where the RLS operator is diagonal or block-diagonal.
    *   **RLS Update/Apply**: Executes the learning logic. The `rls_state` object acts as a stateful accumulator, updating the spectral coefficients based on the `glr_proposal` (Global Learning Rate).
    *   **Residual Injection**: Implements a controlled additive residual connection (`state[i] += out_spec[i] * 0.1f`), ensuring the original signal anchor is preserved, preventing gradient vanishing during spectral modulation.

#### 3. AVX-512 & Physics Architecture Alignment
*   **Zero-Cost/Heapless Execution**: The use of `alignas(64)` stack-allocated buffers (`x_spec`, `y_spec`, `out_spec`) ensures cache-line alignment (64 bytes) for AVX-512 load/store operations. This eliminates heap fragmentation and dynamic allocation latency, critical for real-time inference.
*   **SIMD Vectorization**: The `fwht_inplace` and the final residual loop are designed for 512-bit wide registers (`zmm` registers). By processing 16 `float` elements per instruction, the architecture achieves near-peak throughput for the Hadamard butterfly operations.
*   **Memory Locality**: The implementation avoids pointer chasing by utilizing contiguous memory blocks, maximizing L1/L2 cache hit rates. The `d_model` constraint (max 2048) is specifically chosen to fit within the L1 cache, ensuring the entire spectral transformation pipeline remains "on-chip."

#### 4. Architectural Contribution
This file serves as the bridge between the **Neural Engine's state memory** and the **Spectral Operator**. By keeping the logic pure C++ and stack-bound, it allows the compiler to generate optimal assembly (e.g., `vaddps`, `vmovaps`) without the overhead of virtual dispatch or object-oriented abstraction, maintaining the "Physics-First" performance mandate of the CENTAUR engine.

---

## File: `nn\core\spectral\spectral_logic.hpp`

### Architectural Analysis: `spectral_logic.hpp`

This header defines the interface for the **Spectral Logic Layer**, a critical component of the CENTAUR Neural Engine’s frequency-domain state evolution. It facilitates the transition from spatial-temporal activations to spectral updates, enabling non-stationary system modeling via recursive least squares (RLS).

#### Core Purpose
The module implements a **Spectral-Domain Adaptive Filter** that operates directly on the `state` buffer. By leveraging Kronecker-factored RLS, it approximates the inverse Hessian of the loss surface with minimal memory overhead, allowing the engine to adapt to dynamic input distributions without requiring backpropagation through time (BPTT).

#### Critical Function: `spectral_logic_step`
*   **Role:** Acts as the primary execution bridge between the ACT (Adaptive Computation Time) loop and the spectral state.
*   **Mechanism:** 
    *   **Frequency Transformation:** Maps spatial activations into the spectral domain (typically via optimized FFT/DCT kernels).
    *   **RLS Update:** Updates the `KroneckerRLSState` using the `glr_proposal` (Generalized Learning Rate).
    *   **Spectral Correction:** Applies a learned gain matrix to the state, effectively performing a preconditioner-based update to the neural weights.

#### AVX-512 & Zero-Cost Integration
*   **Heapless Execution:** The `KroneckerRLSState` is designed for stack-allocated or pre-allocated memory pools, ensuring zero dynamic allocation during the inference/training loop.
*   **SIMD Alignment:** The `float*` pointers are expected to be 64-byte aligned. The implementation utilizes `_mm512_fmadd_ps` and `_mm512_load_ps` to process 16-wide float vectors, maximizing throughput for the Kronecker product operations.
*   **Zero-Cost Abstraction:** By passing the `rls_state` by reference and utilizing compile-time `d_model` sizing (where applicable), the architecture eliminates pointer indirection and virtual dispatch overhead, allowing the compiler to inline the spectral update directly into the hot path of the neural engine.
*   **Physics-Informed Constraints:** The spectral correction step enforces stability constraints on the state update, ensuring the system remains bounded—a requirement for the engine's underlying physics-based convergence guarantees.

---

## File: `nn\core\vision\scanner.cpp`

### Architectural Analysis: `scanner.cpp`

This module implements the **Vision Stage 1** wavefront processing for the CENTAUR Neural Engine. It prioritizes cache-locality and deterministic execution, mapping high-dimensional state-space models (SSM) and spatial convolutions directly onto AVX-512 registers to minimize memory bus pressure.

#### 1. `ssm2d_scan` (Wavefront State Propagation)
*   **Purpose:** Executes a 2D causal scan across a feature map. By iterating via wavefront diagonals ($k = r + col$), it maintains strict temporal dependency ordering while maximizing spatial reuse of the hidden state $h$.
*   **AVX-512 Optimization:** Uses `_mm512_fmadd_ps` for fused multiply-add operations, effectively performing the SSM recurrence $h_t = A \cdot h_{t-1} + B \cdot x_t$ in a single cycle per vector.
*   **Architectural Constraint:** The implementation avoids heap allocation by relying on stack-allocated `zero_buf` and pointer arithmetic. The branchless logic for the initial state ($idx=0$) ensures the pipeline remains un-stalled by conditional jumps.
*   **Bottleneck:** The `_mm512_reduce_add_ps` operation is a horizontal reduction; in future iterations, this should be replaced with a masked store or a dot-product intrinsic to avoid scalarization of the output $y$.

#### 2. `dwconv2d_3x3` (Spatial Feature Extraction)
*   **Purpose:** Performs depthwise spatial filtering. This is the primary feature extractor for the vision backbone, designed to operate on contiguous memory blocks.
*   **AVX-512 Optimization:** Employs a 16-wide float vectorization strategy. By unrolling the $3 \times 3$ kernel loop, it minimizes the overhead of boundary condition checks (`iy`, `ix` bounds).
*   **Zero-Cost Strategy:** The function operates on raw pointers with `__restrict` qualifiers, allowing the compiler to perform aggressive alias analysis and register promotion. This eliminates the need for intermediate buffers or object-oriented abstractions that typically introduce heap-related latency.

#### 3. Systems Integration
*   **Memory Layout:** The code assumes a `[H, W, C]` layout where $C$ is the innermost dimension. This is critical for the AVX-512 load/store efficiency, as it ensures that the 512-bit registers are always filled with contiguous channel data.
*   **Heapless Design:** By avoiding `std::vector` or dynamic allocation within the hot loops, the engine maintains a deterministic memory footprint. This is essential for the CENTAUR engine's requirement of running within fixed-size SRAM/L1 cache partitions.
*   **Portability:** The `#if defined(__AVX512F__)` guard provides a fallback to AVX2, ensuring the engine remains functional on legacy hardware while reserving the full 512-bit width for high-throughput inference nodes.

#### Critical Observations
*   **Instruction Density:** The use of `_mm512_loadu_ps` (unaligned) is a deliberate trade-off to support arbitrary feature map dimensions without requiring padding, which would otherwise waste precious cache space.
*   **Performance Note:** The `ssm2d_scan` function currently performs a scalar `y` write. To reach peak throughput, the output $y$ should be vectorized to match the $h$ state update, potentially using a wider output buffer to allow for 512-bit stores.

---

## File: `nn\core\vision\scanner.hpp`

### Architectural Analysis: `nn/core/vision/scanner.hpp`

This module serves as the **Vision Front-End** for the CENTAUR Neural Engine, designed for high-throughput spatial feature extraction. It enforces a strict **NHWC memory layout**, which is critical for aligning 512-bit vector registers with channel-contiguous data, minimizing cache-line splits during depthwise operations.

#### 1. `dwconv2d_3x3` (Spatial Feature Extraction)
*   **Mechanism:** Implements a depthwise convolution where each channel is processed independently.
*   **AVX-512 Strategy:** By utilizing `_mm512_fmadd_ps` (FMA) on 16-float blocks, the implementation achieves peak arithmetic intensity. The `__restrict` qualifiers enable the compiler to perform aggressive load-store motion and avoid aliasing checks, facilitating a "heapless" execution path where the kernel operates directly on pre-allocated scratchpad buffers.
*   **Performance Goal:** Minimizing latency in the vision pipeline by keeping the working set within L1/L2 cache, avoiding the overhead of dynamic memory allocation.

#### 2. `ssm2d_scan` (Temporal/Spatial State Transition)
*   **Mechanism:** Performs a linearized state-space transformation. It maps 2D spatial grids into a 1D sequence, allowing the engine to model long-range dependencies via recurrent state updates.
*   **AVX-512 Strategy:** The scan operation is inherently sequential (data dependency on `h`). To maintain performance, the architecture utilizes **prefix-sum vectorization** or **parallel scan algorithms** (e.g., Blelloch scan) adapted for AVX-512. This allows the engine to process multiple state dimensions in parallel across the 512-bit register width.
*   **Performance Goal:** Enabling "Zero-Cost" state transitions by performing in-place updates on the `h` buffer, eliminating the need for intermediate tensor copies or heap-allocated state history.

#### 3. Systems Integration (The "Zero-Cost" Philosophy)
*   **Memory Layout:** The NHWC format is chosen specifically to ensure that the `C` dimension (channels) is always aligned with the 512-bit vector width. This allows the `Scanner` to perform vector loads/stores without `vperm` or `vgather` instructions, which are typically performance bottlenecks.
*   **Heapless Execution:** By passing `ScannerConfig` by value and relying on caller-provided pointers, the module enforces a **static memory footprint**. This is essential for the CENTAUR engine's deterministic execution model, ensuring that the vision pipeline operates within a fixed memory budget, suitable for real-time embedded or high-frequency trading environments.
*   **Instruction-Level Parallelism (ILP):** The separation of spatial convolution and state-space scanning allows the compiler to interleave instructions from both functions, maximizing the utilization of the AVX-512 execution ports (FMA, Shuffle, and Load/Store units) simultaneously.

---

## File: `nn\core\vision\spectral_pruner.cpp`

### Architectural Analysis: `spectral_pruner.cpp`

The `SpectralPruner` implements a geometric proxy for token importance, replacing expensive SVD-based dimensionality reduction with a single-pass power iteration. It serves as a high-throughput filter for the CENTAUR Neural Engine’s vision pipeline, prioritizing tokens that contribute most to the principal component of the latent space.

#### 1. Core Mechanism: Geometric Proxy
The algorithm approximates the leading singular vector by computing the mean vector of the input patch manifold. By projecting individual tokens onto this mean vector, it derives a scalar importance score (L2-norm proxy). This reduces the complexity of token selection from $O(N^2D)$ to $O(ND)$, enabling real-time pruning within the inference loop.

#### 2. AVX-512 Implementation Strategy
*   **Vectorized Accumulation:** The mean vector calculation utilizes `_mm512_add_ps` to perform 16-wide floating-point additions per cycle. This maximizes throughput on the FMA units, effectively saturating the memory bandwidth of the cache lines containing `x_patches`.
*   **Fused Multiply-Add (FMA) Scoring:** The scoring phase leverages `_mm512_fmadd_ps` to compute dot products. By accumulating into a 512-bit register and utilizing `_mm512_reduce_add_ps` only at the tail of the vector, the implementation minimizes horizontal dependency stalls.
*   **Memory Alignment:** The code assumes unaligned loads (`_mm512_loadu_ps`). While functional, this architecture would benefit from 64-byte alignment of `x_patches` to enable `_mm512_load_ps` and avoid potential cache-line split penalties.

#### 3. Critical Performance Bottlenecks
*   **Heap Allocation:** The current implementation allocates `std::vector<float> v`, `scores`, and `indices` on the heap. In a "zero-cost, heapless" architecture, these should be replaced with stack-allocated buffers or pre-allocated scratchpads provided by the `CENTAUR` memory manager to avoid non-deterministic latency and TLB pressure.
*   **Selection Complexity:** `std::nth_element` is $O(N)$ on average, but the subsequent `std::sort` on the active indices introduces $O(K \log K)$ overhead. For small $K$, a partial sort or a bitmask-based selection would be more efficient.

#### 4. Integration with CENTAUR Physics Architecture
This module acts as a **spatial filter**. By pruning low-energy tokens before they reach the attention mechanism, it reduces the effective sequence length $N$, providing a quadratic reduction in subsequent compute cycles. To achieve the "pure C++" goal, the `std::vector` dependencies must be refactored into a fixed-size `std::array` or a span-based scratchpad interface to ensure deterministic memory access patterns and cache-locality.

---

## File: `nn\core\vision\spectral_pruner.hpp`

### Architectural Analysis: `SpectralPruner`

The `SpectralPruner` serves as the **Phase 11 dimensionality reduction engine** within the CENTAUR pipeline. It replaces traditional, computationally expensive Singular Value Decomposition (SVD) with a SIMD-accelerated Power Iteration method to identify the principal components of input patches, effectively pruning low-variance tokens before they reach the attention mechanism.

#### Core Purpose
*   **Complexity Reduction:** Transforms $O(N^3)$ spectral analysis into $O(N)$ linear estimation.
*   **Information Density:** Maximizes the signal-to-noise ratio by discarding tokens that contribute minimally to the spectral variance of the feature map.
*   **Hardware Alignment:** Designed for AVX-512 FMA (Fused Multiply-Add) throughput, ensuring the pruning decision is made within the cache-coherent execution window.

#### Critical Functions
*   **`prune(...)`**: The primary execution kernel. It consumes a contiguous `std::span` of patch data and populates an index buffer. By operating on raw memory spans, it avoids heap allocations, maintaining the "zero-cost" requirement.
*   **`Config`**: A POD (Plain Old Data) structure that defines the static geometry of the input tensor, allowing the compiler to unroll loops and optimize register pressure for specific `d_model` widths.

#### Integration with CENTAUR Architecture
*   **Heapless Execution:** By utilizing `std::span` and stack-allocated or pre-allocated buffers, the pruner avoids runtime memory management, preventing non-deterministic latency spikes during inference.
*   **AVX-512 Vectorization:** The implementation is architected to leverage 512-bit ZMM registers. By processing 16 `float` elements per instruction, the Power Iteration converges in a constant number of cycles, ensuring deterministic execution time.
*   **Cache Locality:** The design assumes a flat memory layout, facilitating streaming SIMD loads that saturate the L1 cache bandwidth, critical for the high-throughput requirements of the CENTAUR Neural Engine.
*   **Zero-Cost Abstraction:** The class structure provides a clean interface for the compiler to inline the pruning logic directly into the vision pipeline, eliminating function call overhead and enabling aggressive inter-procedural optimization.

---

## File: `nn\scratch.cpp`

### Architectural Analysis: `nn\scratch.cpp`

This module serves as a **compile-time reflection and harness generator** for the CENTAUR Neural Engine’s kernel execution pipeline. It abstracts the boilerplate required to inject memory-aligned, restricted-pointer buffers into AVX-512 compute kernels.

#### Core Purpose
The system automates the instantiation of kernel arguments, ensuring that pointer aliasing constraints (`__restrict`) are propagated to the compiler’s optimizer. By leveraging `std::tuple` and `std::index_sequence`, it bridges the gap between generic kernel signatures and the specific memory-layout requirements of the AVX-512 execution units.

#### Critical Components

*   **`ArgGen<T>` Specializations**: Acts as a RAII-based memory provider. By specializing on pointer qualifiers, it forces the compiler to treat kernel inputs as distinct, non-overlapping memory regions, which is a prerequisite for effective **AVX-512 auto-vectorization** and loop unrolling.
*   **`invoke_helper`**: A variadic template unpacker. It performs a parameter pack expansion to map the `ArgGen` tuple into the function signature of the target kernel. This eliminates manual argument marshalling, reducing the risk of pointer-type mismatches in high-performance paths.
*   **`run_auto_benchmark`**: The entry point for the harness. It performs type-erasure of the kernel signature, allowing the engine to dynamically dispatch kernels while maintaining strict type safety and alignment guarantees at the call site.

#### Contribution to CENTAUR Architecture
*   **Zero-Cost Abstraction**: The template metaprogramming approach ensures that all argument resolution occurs at compile-time. The generated machine code is equivalent to a direct function call with raw pointers, incurring zero runtime overhead for the harness itself.
*   **Memory Alignment & Vectorization**: By enforcing `__restrict` via the `ArgGen` template, the architecture guarantees that the compiler can safely assume no pointer aliasing. This is critical for the AVX-512 backend to generate `vmovaps` (aligned move) instructions rather than slower, unaligned loads.
*   **Heapless Strategy**: While the current implementation uses `new[]` for demonstration, the architecture is designed to be swapped with a **static arena allocator** or a **scratchpad memory buffer** (e.g., L1-cached memory pools). The `ArgGen` pattern allows for a seamless transition to stack-allocated or pre-allocated memory without modifying the kernel invocation logic, maintaining the "heapless" requirement for real-time neural inference.

---

