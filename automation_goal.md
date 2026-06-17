# Local 24/7 Goal: Weight-Space Trading Engine

Build an anytime model-optimization system that treats an existing LLM as an environment.

The optimizer should not distill a student model and should not freely rewrite arbitrary weights. It should learn and search for safe compression decisions on existing model tensors.

## Core Thesis

Use tensor indicators as state, discrete quantization choices as actions, and a reward based on memory saved versus behavioral damage.

State examples:
- tensor role and layer index
- activation variance
- outlier ratio
- sparsity
- norm and max/median ratio
- quantization reconstruction error

Actions:
- keep high precision
- quantize to Q8
- quantize to Q6
- quantize to Q5
- quantize to Q4

Reward:

```text
reward = RAM_saved * alpha - KL_divergence * beta
```

The first useful artifact is not a DQN. It is a reliable reward oracle and an optimization atlas.

## Current Small Goal

Build a causal atlas for one model tensor:

1. Extract tensor statistics.
2. Run calibration prompts.
3. Cache baseline logits.
4. Simulate exact deployment quantization.
5. Measure KL divergence and perplexity drift.
6. Record which tensor regions are fragile or compressible.
7. Produce a mixed-precision atlas that beats naive full-model Q4/Q5.

## Guardrails

- Use an Agentless workflow: localization, proposal, validation.
- Avoid CrewAI, LangGraph, LangChain, or other heavy orchestration layers for the 24/7 local worker.
- Keep the local model as a fixed workflow component, not a free tool-using agent.
- Do not trust benchmark-shaped demos without behavior-preservation checks.
- Do not optimize against CTest alone unless tests are registered and meaningful.
- Do not apply autonomous code rewrites without explicit human review.
- Prefer narrow experiments that falsify the idea quickly.
- Preserve user changes in the repository.
