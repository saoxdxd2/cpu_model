# Volume III: Bit-Level Intelligence and Execution Streamlining

The CENTAUR Architecture has aggressively streamlined its execution graph by removing legacy, high-level abstractions like `multimodal_engine` and `encoding` directories. This volume details the shift towards a pure, unified binary neural engine.

## 1. Deprecation of Multimodal Encoders

Previous iterations utilized a `SiliconEncoder` and complex Multimodal Vision-Code fusion. To finalize the physical-level optimization of the CENTAUR execution graph, these modules were **removed**.

### 1.1 Pure Symbolic Grounding
By pruning the `encoding` and `multimodal_engine` components, we enforce a strictly static memory layout that operates directly on raw, cache-aligned buffers. The engine now perceives the environment natively through its pure binary representations, eliminating cross-modal overhead.

## 2. Alphabet-Primitive Deduction (NPP)

Instead of using a massive token vocabulary, CENTAUR uses a **Primitive Codebook**.

### 2.1 Next-Primitive Prediction (NPP)
The model is trained on the raw character combinations of millions of lines of source code. By predicting the next byte, the model develops an internal **Combinatorial Logic**:
*   **Semantic Emergence**: Through millions of NPP cycles, the model deduces that the sequence `f-u-n-c-t-i-o-n` represents a logical block.
*   **Efficiency**: Character-level prediction is massively less memory-intensive than word-piece tokenization.

## 3. The Saliency Tokenizer

The `SaliencyTokenizer` maps continuous thoughts back to discrete "Thought Anchors" without relying on dynamic allocations (`std::vector` or `std::shared_ptr`).
*   **Discretization**: Maps the latent vector to a discrete concept index in a highly compressed dictionary.
*   **Stable Recall**: This bridge allows the CENTAUR intelligence to remain contextually stable over long conversations or complex coding tasks, executing entirely on stack-allocated or static memory.

---
*Next Volume: [Silicon Swarm & Scaling](SILICON_SWARM.md)*
