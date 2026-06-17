import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

def nca_attention_forward(self, hidden_states, attention_mask=None, layer_past=None, **kwargs):
    """
    Mock-up of the 'NCA Graph-Curve / Recurrent' architecture.
    Instead of doing O(N^2) Softmax(Q * K^T) * V, it treats the attention weights 
    as an O(1) recurrent projection or linear kernel, simulating the 'heapless SIMD' path.
    """
    c_attn = self.c_attn(hidden_states)
    query, key, value = c_attn.split(self.split_size, dim=2)

    # Standard attention would be:
    # attn_weights = torch.matmul(query, key.transpose(-1, -2))
    # attn_weights = attn_weights / math.sqrt(key.size(-1))
    # attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    # attn_output = torch.matmul(attn_weights, value)

    # NCA 'Recurrent State' Simulation (O(1) memory, no KV cache across N):
    # We simulate collapsing the sequence dimension by doing a simple moving average 
    # or just using local projection, representing the 'static kernel' claim.
    # Here, we just project the value directly, simulating an identity mapping 
    # for the attention score, which is O(1) w.r.t sequence length.
    
    # Just project V and add a simple recurrent decay (mocking 'recursive wavefront calculus')
    attn_output = torch.zeros_like(value)
    state = torch.zeros_like(value[:, 0:1, :])
    
    # Simulate a linear recurrent path (like an SSM or RWKV without the proper weights)
    alpha = 0.9
    for i in range(hidden_states.size(1)):
        # Update recurrent state
        state = alpha * state + (1 - alpha) * value[:, i:i+1, :]
        attn_output[:, i:i+1, :] = state

    attn_output = self.c_proj(attn_output)
    return attn_output, None

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def evaluate_model():
    print("="*60)
    print(" KILL-TEST: True Weight-Preserving NCA Eval")
    print("="*60)

    model_id = "gpt2" # Fast, trained model for testing
    print(f"[1] Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load original baseline model
    model_baseline = AutoModelForCausalLM.from_pretrained(model_id)
    model_baseline.eval()

    # Load a second copy for NCA manipulation
    model_nca = AutoModelForCausalLM.from_pretrained(model_id)
    model_nca.eval()

    print("[2] Hijacking attention layers with 'NCA Recurrent Wavefront' logic...")
    # Replace standard attention with our O(1) recurrent logic
    for block in model_nca.transformer.h:
        block.attn.forward = nca_attention_forward.__get__(block.attn, type(block.attn))

    # Test Prompt
    prompt = "The future of CPU inference is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print(f"\n[3] Testing prompt: '{prompt}'")
    print("    Sequence length:", input_ids.shape[1])

    with torch.no_grad():
        # Baseline
        out_baseline = model_baseline(**inputs, labels=input_ids)
        loss_baseline = out_baseline.loss.item()
        ppl_baseline = math.exp(loss_baseline)
        logits_baseline = out_baseline.logits

        # NCA
        out_nca = model_nca(**inputs, labels=input_ids)
        loss_nca = out_nca.loss.item()
        ppl_nca = math.exp(loss_nca)
        logits_nca = out_nca.logits

    print("\n[4] Results:")
    print(f"    Baseline Perplexity: {ppl_baseline:.4f}")
    print(f"    NCA Perplexity:      {ppl_nca:.4f}")
    
    print("\n[5] Logit Comparison (First Token, First 5 values):")
    print(f"    Baseline: {logits_baseline[0, 0, :5].tolist()}")
    print(f"    NCA:      {logits_nca[0, 0, :5].tolist()}")

    # Cosine Similarity of Logits
    cos_sim = torch.nn.functional.cosine_similarity(
        logits_baseline.flatten().unsqueeze(0), 
        logits_nca.flatten().unsqueeze(0)
    ).item()
    print(f"\n    Logit Cosine Similarity: {cos_sim:.4f}")

    print("\n" + "="*60)
    if cos_sim < 0.9:
        print("[VERDICT] FAILURE. The NCA 'zero-cost transition' fundamentally")
        print("          breaks the pre-trained weights. Perplexity exploded.")
        print("          This architecture cannot reuse transformer weights.")
    else:
        print("[VERDICT] SUCCESS. The weights transferred successfully.")
    print("="*60)

if __name__ == "__main__":
    evaluate_model()
