# NCA — Aether Dashboard & AI IDE Roadmap

> **Status**: Core Silicon Engine Verified.
> **Current Phase**: Architectural Hardening & UX Design.

## 1. Unified Aether Dashboard
A high-fidelity GUI (likely React/Electron) to control the hardware-saturated engine.

### A. Model Selection Page
*   **Active Model**: Current Gemma-4 Silicon instance.
*   **Model Hub**: One-click download/adoption of new GGUF models via the `gguf_translator`.
*   **Topology Viewer**: Real-time visual map of the engine's "Silicon Wiring" (Expert activation, GLR wavefront).

### B. Scaling & Training Page
*   **Drag-and-Drop Training**: Drop a folder or file to trigger the `ai_ide_trainer` (Alphabet-Level).
*   **Fine-Tuning Dashboard**: Real-time loss/entropy curves.
*   **Weight Export**: One-click `.pt` export of tuned weights.

### C. Tool & Peripheral Registry
Dynamic "Wiring" of agentic capabilities based on model needs:
*   **Percepts**: Enable/Disable Keyboard, Mouse, Screenshots, Screen Recording.
*   **MCPs (Model Context Protocols)**: Connect external APIs and tools.
*   **Environment**: Switch between `TacticalGridEnv` and `VSCodeEnv`.

## 2. Multi-Room Flexibility
Future-proofing the engine for diverse creative tasks.

### A. The Video Room
*   **AI Video Editing**: Map agentic actions to timeline operations (cut, transition, color-grade).
*   **Frame-Saliency**: Using `VisionEncoder` to identify key frames.

### B. The Creative Room
*   **AI Image Generation**: Bridge to external diffusion models using the NCA's "Thought Vectors" as a seed.
*   **Image Understanding**: Deep grounding of generated visual data.

## 3. Current Code Hardening (Ongoing)
*   **Zero-Allocation Enforcement**: Remove all `std::vector` and heap usage in the hot path.
*   **Circular Dependency Wipe**: Ensure clean modular boundaries.
*   **O(1) Efficiency**: Final optimization of the Saliency-Driven Multi-expert System (SDMS).

---
*Target: Turn the Aether Pipeline into a universal "Silicon Intelligence" for every developer task.*
