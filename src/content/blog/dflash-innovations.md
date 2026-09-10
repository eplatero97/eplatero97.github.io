---
title: "How D-Flash Speeds Up Inference"
pubDate: 2026-09-10
description: "A walkthrough of D-Flash — the block-diffusion speculative decoder — and how its KV cache injection works in vLLM."
category: "AI"
tags: ["AI", "speculative-decoding", "vllm", "inference", "llm"]
cover: "/assets/images/dflash_kv_cache_injection.png"
---

# D-Flash?

D-Flash is a block diffusion model that generates $\gamma$ draft tokens in parallel. This is similar to the Medusa technique, but instead of using a simple MLP, D-Flash uses the expressive power of the attention mechanism. Generating tokens in parallel while staying expressive are two of the qualities that have made D-Flash so popular, and have helped inspire new ideas like [D-Spark](https://arxiv.org/pdf/2607.05147) and [D-Flash 2](https://inco.ai/blog/dflash2/).

## Is D-Flash Really Flash?

D-Flash has two primary innovations: (i) Draft KV cache injection and (ii) model architecture. Both of these innovations essentially parallelize bottlenecks in traditional speculative decoding techniques. 

A useful annotation that contrasts these innovations with previous methods is:

$$
\begin{aligned}
\text{Vanilla draft:}\quad
& q_\phi(\text{next draft token} \mid \text{prefix}, \text{bonus token}, \text{previous draft tokens}) \\
\text{EAGLE:}\quad
& q_\phi(\text{next draft token} \mid \text{prefix}, \text{bonus token}, \text{previous draft tokens}, \text{target hidden states}) \\
\text{D-Flash:}\quad
& q_\phi(\text{draft block} \mid \text{bonus token}, \text{MASK block}, KV_{\text{DFLASH}})
\end{aligned}
$$

$$
\text{where } KV_{\text{DFLASH}}=\mathcal{A}(\text{target hidden states})
$$

Here, the terms are:

- $q_\phi$ — the draft proposal distribution.
- $\text{prefix}$ — the prompt + verified tokens.
- $\text{target hidden states}$ — the extracted hidden states from the target model prefix.
- $\mathcal{A}$ — the complete KV-injection algorithm that supplies the context to the draft model.

The key difference is that Vanilla and EAGLE still propose the $\text{next draft token}$ autoregressively, while D-Flash proposes a whole $\text{draft block}$ from the bonus token, MASK positions (length $\gamma$), and injected DLM KV memory.

We'll take each innovation in turn, starting with the KV cache injection.

## KV Cache Injection

Like EAGLE, D-Flash conditions its speculations on a subset of hidden states from the target model. However, instead of feeding these hidden states as inputs along with token embeddings, the compressed projection of these target hidden states is injected directly into the KV cache of the draft model... *mostly* IN PARALLEL!

What this means is that from those compressed projections, a single fused projection produces every DLM layer's KV entries at once, bypassing the traditional layer-stacked architecture of LLMs (don't you wish we could do this for all LLMs?). This parallelization keeps the KV update and its cost during time-to-first-token (TTFT) minimal.

Let's look at how this is done in vLLM `0.23.0`.

## Rich Embeddings + Parallel KV Projection Implementation

During the target model's forward pass, D-Flash extracts hidden states from multiple target-model layers. For each token, those hidden states are concatenated and then compressed into a single "rich embedding." This embedding has proven to be expressive enough for the draft model to reconstruct all of its own KV-cache entries from it.

For example, if three target-model layers are extracted, a token might start as three vectors of size `H_tlm`, or `[3 * H_tlm]` after concatenation. The auxiliary FC maps that wide representation into `[H_dlm]`, the draft model's hidden size. The result is the compact hidden state that feeds the KV-injection path.

Here is the whole KV-injection process at a high level:

![D-Flash KV cache injection](/assets/images/dflash_kv_cache_injection.png)

*The target model supplies the hidden states; the DLM's layers supply the KV projections; the green block writes the resulting KV tensors into the DLM cache.*

The figure shows one of D-Flash's most clever tricks: after attaining the compact hidden states, it re-uses the draft model KV-weight projections to transform the rich embedding into its layer-corresponding KV entries representation in parallel. This means that the draft KV-projections have been trained to accept inputs from the compressed target model hidden states and the draft model's own hidden state inputs to the attention layer. 

In vLLM, this logic lives in `precompute_and_store_context_kv` in `vllm/model_executor/models/qwen3_dflash.py`. The fused KV-projection weights are built once after loading the DLM weights. Each layer contributes the KV rows of its QKV projection — `qkv_proj.weight[q_size:]` removes the query rows — and those slices are concatenated into one large matrix.

```python
# 1. RMSNorm on input hidden states — shared norm before the GEMM.
ops.rms_norm(
    normed_context_states,
    context_states,
    self._hidden_norm_weight,
    self._rms_norm_eps,
)

# 2. ONE fused GEMM for all DLM layers.
#    self._fused_kv_weight = cat([qkv_proj.weight[q_size:] for each layer])
#    shape: [H_dlm, 2 × L × nkv × head_dim] — built once at weight-load time
all_kv_flat = F.linear(normed_context_states, self._fused_kv_weight)
#    out: [num_ctx, L × 2 × nkv × head_dim]

# 3. Reshape + permute into a layer-major staging layout.
#    This is an intermediate tensor, not the physical paged KV-cache layout.
all_kv = (
    all_kv_flat
    .view(num_ctx, L, 2, nkv, head_dim)
    .permute(2, 1, 0, 3, 4)
    .contiguous()
)
#    staging output: [2, L, num_ctx, nkv, head_dim]
all_k = all_kv[0]  # [L, num_ctx, nkv, head_dim]
all_v = all_kv[1]  # [L, num_ctx, nkv, head_dim]

# 4. Per-layer RMSNorm on K — separate weight per layer.
#    self._k_norm_weights[i] = layers[i].self_attn.k_norm.weight
for i in range(L):
    ops.rms_norm(all_k_normed[i], all_k[i], self._k_norm_weights[i], ...)

# 5. Fused RoPE across ALL layers in one kernel call.
#    Trick: flatten [L, num_ctx, kv] → [L * num_ctx, kv],
#           repeat context_positions L times → [L * num_ctx].
all_k_flat = all_k_normed.view(L * num_ctx, kv_size)
ops.rotary_embedding(context_positions.repeat(L), all_k_flat, None, ...)

# 6. Per-layer cache write; V stored as-is (no norm, no RoPE).
#    Each update receives [num_ctx, nkv, head_dim] for one layer. The attention
#    backend maps those rows into VLLM's paged cache, logically
#    [num_blocks, 2, block_size, nkv, head_dim].
for i in range(L):
    attn.impl.do_kv_cache_update(
        attn, all_k_final[i], all_v[i],
        attn.kv_cache, context_slot_mapping,
    )
```

The first `rms_norm` is shared across all context states before the fused GEMM. The single `F.linear` produces Key and Value projections for every context token and every DLM layer at once. Instead of invoking one projection per layer, vLLM uses the block matrix built from all of those layer-specific KV weights.

The reshape and permutation turn the flat output into `[2, L, num_ctx, nkv, head_dim]`. The leading dimension separates K from V; the next dimension identifies the DLM layer. This is what lets the rest of the method process every layer's cache entries without running the DLM over the prompt.

After the split, the Key matrix receives a separate RMSNorm for each layer and then RoPE; the Value matrix receives neither. The implementation flattens the layer and token dimensions so one rotary-embedding kernel handles all of them. Finally, `do_kv_cache_update()` writes each layer's processed Key and untouched Value into the appropriate cache slots.

At this point, the DLM's cache entries for the target-model context are populated. The DLM never needed to run a forward pass over those prompt tokens. It only needed the target model's hidden states, the fused projection, and the cache update.

## What Happens After Injection

Once the context portion of the cache is populated, the draft model runs its block-parallel speculative-decoding forward pass. The bonus token and MASK query tokens produce their own KV entries, which are written into the later cache positions and used by the draft model's attention. The draft model computes logits for the bonus position too, but vLLM samples only the MASK-position logits.

After verification, the "rich embedding → giant KV matmul" path runs again for the updated committed prefix. It replaces the DLM's KV entries for the newly accepted tokens with target-derived ones; the rejected tokens need no special handling, since their slots are simply overwritten by the next draft's forward pass.

## The Key to Injection: Synchronizing the DLM with the Target

The key idea is that KV injection re-anchors the DLM to the target's committed prefix before every draft pass. Turning the target's selected hidden states into DLM KV memory is the synchronization point: once the target verifies a block, the next draft starts from the updated target prefix rather than the DLM's previous proposal.

Each round has two cache sources. The committed context is target-derived, while the DLM writes its own provisional KV entries for the bonus token and masked draft slots — that provisional suffix is the DLM's query-derived KV. The target then resolves it by accepting a prefix and discarding the rest before the DLM is re-anchored.

```text
┌────────────────────────────────────────────────────────────┐
│ 1. COMMITTED TARGET PREFIX                                 │
│                                                            │
│ tokens: [x₀ ... xᵢ₋₁]                                      │
│ target hidden states: [h₀ ... hᵢ₋₁]                        │
└─────────────────────────────┬──────────────────────────────┘
                              │ project target hidden states
                              │ into DLM KV
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 2. INJECT TARGET-DERIVED KV                                │
│                                                            │
│ DLM KV[0..i-1] ← target-derived KV                         │
└─────────────────────────────┬──────────────────────────────┘
                              │ anchor + masked draft slots
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 3. PARALLEL DLM DRAFT                                      │
│                                                            │
│ input_ids: [bᵢ, MASKᵢ₊₁, MASKᵢ₊₂, MASKᵢ₊₃, MASKᵢ₊₄]        │
│                                                            │
│ DLM KV[i..i+4] ← provisional draft-derived KV              │
│ discarded logit: bonus-position logit at i                 │
│ usable logits: [dᵢ₊₁, dᵢ₊₂, dᵢ₊₃, dᵢ₊₄]                    │
└─────────────────────────────┬──────────────────────────────┘
                              │ four proposed tokens
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 4. TARGET VERIFICATION                                     │
│                                                            │
│ proposed: [dᵢ₊₁, dᵢ₊₂, dᵢ₊₃, dᵢ₊₄]                         │
│           [ ✓,  ✓,  ✗,  ✗ ]                                |
│                                                            │
│ next target bonus token: bᵢ₊₃                              │
│ committed: [x₀ ... xᵢ₋₁, bᵢ, dᵢ₊₁, dᵢ₊₂, bᵢ₊₃]             │
│ rejected:  [dᵢ₊₃, dᵢ₊₄]                                    │
└─────────────────────────────┬──────────────────────────────┘
                              │ discard rejected suffix
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 5. REFRESH AND RE-ANCHOR                                   │
│                                                            │
│ keep accepted tokens; truncate provisional draft KV        │
│ recompute target hidden states for the updated prefix      │
│ inject refreshed target-derived DLM KV                     │
│                                                            │
│ → draft the next block                                     │
└────────────────────────────────────────────────────────────┘
```

The diagram's two cache regions are the essential distinction. Boxes 1, 2, and 5 contain target-derived KV for the committed prefix; box 3 adds DLM-generated KV only for the current proposal. Although the target computes verification logits for all four drafts in parallel, the first mismatch ends the accepted draft prefix, so every later provisional entry is discarded. The next bonus token is the target model's first token at the mismatch position, or the target token after the final draft token if all drafts are accepted.

## What the DLM Looks Like

The KV-injection path is novel, but the DLM itself is still a small Qwen3-style decoder. Its architecture is special in three ways.

First, it has only a few layers — typically five, rather than a single layer like EAGLE tends to deploy. This keeps the drafting model lightweight while still giving it more expressive power than a single attention layer. The fused KV projection is what makes those extra layers affordable during context setup.

Second, D-Flash uses non-causal attention during the draft block. This sounds fancy, but the intuition is simple: bidirectional just means each query can attend across all context tokens plus the newly generated block entries — both left and right — instead of the usual left-to-right lower triangular matrix.

Finally, the draft positions begin as MASK tokens and are processed together with the bonus token. The MASK embeddings start out identical, but positional information and attention to the injected context make each position's hidden state different. One forward pass can therefore produce a different prediction for every draft position. This is block-parallel prediction, not an autoregressive shift inside the block: the logit at a MASK position predicts that position's token, and the logit at the bonus-token position is ignored for speculative proposals.

## Does It Actually Deliver?

Finally, let's get to performance. All of the innovations above buys two things: the fused KV projection keeps TTFT low (no layer-by-layer forward pass over the prefix), and the block-parallel draft keeps per-round cost nearly flat in $\gamma$. Compare that to an autoregressive drafter, whose cost grows with every proposed token ($T_{\text{draft}} \propto \gamma$). The [D-Flash paper](https://arxiv.org/abs/2602.06036) reports up to **~6× lossless speedup** on Qwen3-8B, and roughly **2.5× faster than EAGLE-3**, the previous state-of-the-art speculative decoder (reasoning-heavy workloads land around ~4.5×). Here is a representative slice on Qwen3-8B with greedy decoding, as speedup over the vanilla autoregressive baseline:

| Task | Vanilla | EAGLE-3 | D-Flash |
|------|---------|---------|---------|
| GSM8K | 1× | 2.13× | 5.20× |
| MATH-500 | 1× | 2.18× | 6.17× |
| HumanEval | 1× | 2.48× | 5.20× |
| MT-Bench | 1× | 1.94× | 2.79× |
| Alpaca | 1× | 1.88× | 2.27× |

The gap in performance between D-Flash and EAGLE is a result of (i) a longer average accepted length — τ ≈ 6.5 on Qwen3-8B at block size 16, roughly double EAGLE-3's τ ≈ 3 — and (ii) the block-parallel pass keeping the per-round draft cost nearly flat in $\gamma$, so you accept more tokens per round without paying more to draft them. The gains are largest on structured and reasoning-heavy tasks, where the target's next tokens are more predictable, and shrink on open-ended chat (MT-Bench, Alpaca) and at high concurrency, where verification and memory traffic start to dominate.

## What Comes Next?

D-Flash has been a seminal work that quickly spurred new follow-ups like D-Spark and D-Flash 2. In different ways, both of these follow-ups target the multi-modal collision problem with parallel drafters, which states that even though each draft token is plausible, because of their independent block prediction, the resulting draft as a whole may not be cohere, so acceptance quickly decays after the first draft token. If you want to go deeper on the problem, make sure to check out [this write-up](https://www.the-information-bottleneck.com/p/speculative-decoding-from-zero-to) and stay tuned for more!
