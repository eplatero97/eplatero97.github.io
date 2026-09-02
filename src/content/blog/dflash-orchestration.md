---
title: "D-Flash KV Cache Injection"
pubDate: 2026-07-25
description: "D-Flash KV Cache Injection Algorithm"
category: "AI"
tags: ["AI"]
cover: "/assets/images/dflash_kv_cache_injection.png"
---

# D-Flash?

D-Flash is a speculative decoding model that uses a diffusion-like model to generate $\gamma$ draft tokens in parallel. This is similar to the Medusa technique, but instead of using a simple MLP, D-Flash uses the expressive power of the attention mechanism. The ability to generate tokens in parallel while staying expressive are two of the qualities that have made D-Flash so interesting, and have helped inspire new ideas like Gemma 4 Diffusion and D-Spark.

# Is D-Flash Really Flash?

D-Flash has two primary innovations: (i) Draft KV cache injection and (ii) model architecture. Both of these innovations essentially parallelize bottlenecks in traditional speculative decoding techniques. 

A useful annotation that contrasts these innovations with previous speculative decoding methods is:

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
\text{where } KV_{\text{DFLASH}}=\mathcal{A}(e^{\text{target}}_{<i})
$$

Here, $q_\phi$ is the draft proposal distribution. The key difference is that Vanilla and EAGLE still propose the next draft token autoregressively, while D-Flash proposes a whole draft block from the bonus token, MASK positions, and injected DLM KV memory. $\gamma$ is the number of usable draft tokens, and $\mathcal{A}$ denotes the complete KV-injection pipeline: the learned projection, normalization, RoPE processing, and cache write that transform target hidden states into DLM KV memory.

The important distinction is that D-Flash is not autoregressive inside the drafted block. It processes the bonus token and the $\gamma$ MASK slots in one pass, then uses only the MASK-position logits as draft proposals. In vLLM terms, a block size of $B$ gives $\gamma=B-1$ usable draft tokens because the bonus-token output is just an anchor, not a proposal.

The key shift in D-Flash is that target-side information conditions drafting through injected KV memory while the bonus token and $\gamma$ MASK positions are processed together, with positional (RoPE) encoding separating each slot.

We will discuss each innovation in order but will primarily review (i).

## KV Cache Injection

Like EAGLE, D-Flash conditions its speculations on a subset of hidden states from the target model. However, instead of feeding these hidden states as inputs along with tokens, the compressed projection of these hidden states is injected directly into the KV cache of the draft model... *mostly* IN PARALLEL!

This is huge. The traditional layer-stacked architecture of LLMs is bypassed and the K/V of each layer can be computed independently (don't you wish we could do this for all LLMs?).

Let's look at how this is done in vLLM 0.23.0.

## Rich Embeddings + Parallel KV Projection Implementation

During the target model's forward pass, D-Flash extracts hidden states from multiple target-model layers. For each token, those hidden states are concatenated and then compressed into a single "rich embedding." This embedding is surprisingly expressive: it contains enough information for the draft model to reconstruct all of its own KV-cache entries.

If three target-model layers are extracted, a token might start as three vectors of size `H_tlm`, or `[3 * H_tlm]` after concatenation. The auxiliary FC maps that wide representation into `[H_dlm]`, the draft model's hidden size. The result is the compact hidden state that feeds the KV injection path.

Here is the whole process at a high level:

![D-Flash KV cache injection during prefill](/assets/images/dflash_kv_cache_injection.png)

*The target model supplies the hidden states; the DLM's layers supply the K/V projections; the green block writes the resulting K/V tensors into the DLM cache.*

The implementation behind the green block is intentionally compact. In vLLM, this logic lives in `precompute_and_store_context_kv` in `vllm/model_executor/models/qwen3_dflash.py`. The fused K/V projection weights are built once after loading the DLM weights. Each layer contributes the K/V rows of its QKV projection — `qkv_proj.weight[q_size:]` removes the query rows — and those slices are concatenated into one large matrix.

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

Let's unpack this. The first `rms_norm` is shared across all context states before the fused GEMM. The single `F.linear` produces K and V projections for every context token and every DLM layer at once. Instead of invoking one projection per layer, vLLM uses the block matrix built from all of those layer-specific K/V weights.

The reshape and permutation turn the flat output into `[2, L, num_ctx, nkv, head_dim]`. The leading dimension separates K from V; the next dimension identifies the DLM layer. This is what lets the rest of the method process every layer's cache entries without running the DLM over the prompt.

After the split, K receives a separate RMSNorm for each layer and then RoPE. The implementation flattens the layer and token dimensions so one rotary-embedding kernel handles all of them. V receives neither operation. Finally, `do_kv_cache_update()` writes each layer's processed K and untouched V into the appropriate cache slots.

At this point, the DLM's cache entries for the target-model context are populated. The DLM never needed to run a forward pass over those prompt tokens. It only needed the target model's hidden states, the fused projection, and the cache update.

## What Happens After Injection

Once the context portion of the cache is populated, the draft model runs its block-parallel speculative-decoding forward pass. The bonus token and MASK query tokens produce their own K/V entries, which are written into the later cache positions and used by the draft model's attention. The draft model computes an output for the bonus token too, but vLLM samples only the MASK-position outputs.

After verification, the "rich embedding → giant KV matmul" path runs again for the updated committed prefix. Its target-derived K/V entries replace the provisional DLM state used only to make the rejected-or-accepted draft proposal.

## The Key to Injection: Synchronizing the DLM with the Target

The key idea is that KV injection anchors the DLM to the target model's committed prefix. Before each draft pass, the adapter turns the target's selected hidden states into K/V memory for every DLM layer. That injected memory is the synchronization point: after the target verifies a block, the next draft begins from the updated target prefix rather than the DLM's previous proposal.

Each round has two cache sources. The committed context is target-derived, while the DLM writes provisional K/V entries for its bonus token, or anchor token, and masked draft slots. The target resolves that provisional suffix by accepting a prefix and discarding the rest before the DLM is re-anchored.

```text
┌────────────────────────────────────────────────────────────┐
│ 1. COMMITTED TARGET PREFIX                                 │
│                                                            │
│ tokens: [x₀ ... xᵢ₋₁]                                     │
│ target hidden states: [h₀ ... hᵢ₋₁]                       │
└─────────────────────────────┬──────────────────────────────┘
                              │ project target hidden states
                              │ into DLM K/V
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 2. INJECT TARGET-DERIVED KV                                │
│                                                            │
│ DLM KV[0..i-1] ← target-derived K/V                        │
└─────────────────────────────┬──────────────────────────────┘
                              │ anchor + masked draft slots
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 3. PARALLEL DLM DRAFT                                      │
│                                                            │
│ input_ids: [bᵢ, MASKᵢ₊₁, MASKᵢ₊₂, MASKᵢ₊₃, MASKᵢ₊₄]      │
│                                                            │
│ DLM KV[i..i+4] ← provisional draft-derived K/V             │
│ discarded output: bonus-position output at i               │
│ usable outputs: [dᵢ₊₁, dᵢ₊₂, dᵢ₊₃, dᵢ₊₄]                 │
└─────────────────────────────┬──────────────────────────────┘
                              │ four proposed tokens
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 4. TARGET VERIFICATION                                     │
│                                                            │
│ proposed: [dᵢ₊₁, dᵢ₊₂, dᵢ₊₃, dᵢ₊₄]                       │
│           [ ✓,  ✓,  ✗,  ✗ ]                                │
│                                                            │
│ next target bonus token: bᵢ₊₃                              │
│ committed: [x₀ ... xᵢ₋₁, bᵢ, dᵢ₊₁, dᵢ₊₂, bᵢ₊₃]           │
│ rejected:  [dᵢ₊₃, dᵢ₊₄]                                   │
└─────────────────────────────┬──────────────────────────────┘
                              │ discard rejected suffix
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 5. REFRESH AND RE-ANCHOR                                   │
│                                                            │
│ keep accepted tokens; truncate provisional draft KV         │
│ recompute target hidden states for the updated prefix       │
│ inject refreshed target-derived DLM K/V                     │
│                                                            │
│ → draft the next block                                     │
└────────────────────────────────────────────────────────────┘
```

The diagram's two cache regions are the essential distinction. Boxes 1, 2, and 5 contain target-derived K/V for the committed prefix; box 3 adds DLM-generated K/V only for the current proposal. Although the target computes verification logits for all four drafts in parallel, the first mismatch ends the accepted draft prefix, so every later provisional entry is discarded. The next target bonus token is then appended to that prefix before the DLM cache is refreshed.

# What the DLM Looks Like

The KV-injection path is unusual, but the DLM itself is still a small Qwen3-style decoder. Its architecture is special in three ways.

First, it has only a few layers — typically one or a small number, rather than the dozens of layers in the target model. This keeps the drafting model lightweight while still giving it more expressive power than a single attention layer. The fused KV projection is what makes those extra layers affordable during context setup.

Second, D-Flash uses bidirectional attention over the injected context. Each query position can attend to the full target-model context instead of only the tokens to its left. This is safe because the DLM is proposing tokens, not making the final decision: the target model verifies the proposal afterward.

Finally, the draft positions begin as MASK tokens and are processed together with the bonus token. The MASK embeddings start out identical, but positional information and attention to the injected context make each position's hidden state different. One forward pass can therefore produce a different prediction for every draft position. This is block-parallel prediction, not an autoregressive shift inside the block: the logit at a MASK position predicts that position's token, and the logit at the bonus-token position is ignored for speculative proposals.

# Why This Keeps TTFT and TPOT Low

If you use a draft model directly, TTFT suffers because the model must run a full forward pass over the entire input sequence to populate its KV cache. Every layer must produce its own K/V projections, and you pay that cost layer by layer.

D-Flash still needs to populate the draft model's KV cache, but it reuses hidden states that the target model has already computed. The stacked per-layer K/V projections are replaced with one fused projection across all context tokens and DLM layers. That matters especially because D-Flash's drafter is typically deeper than an EAGLE-style one-layer proposer.

For an autoregressive drafter, the drafting cost grows with the number of proposed tokens:

$$
T_{\text{draft}}^{\text{AR}} \propto \gamma.
$$

D-Flash instead processes the masked block in one forward pass, so its draft-side cost is only weakly dependent on $\gamma$. This keeps TPOT, or per-round output latency, from being tightly bound to $\gamma$: increasing $\gamma$ can increase the number of tokens produced per round without multiplying the draft cost. In vLLM terms, a D-Flash checkpoint with block size $B$ should use `num_speculative_tokens = B - 1`; for example, `z-lab/Qwen3-4B-DFlash-b16` has $B=16$, so its vLLM example uses `num_speculative_tokens = 15`. Realized end-to-end latency still depends on verification cost, memory traffic, and how many drafted tokens are accepted.
