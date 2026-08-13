---
title: "D-Flash KV$ Injection"
pubDate: 2026-07-25
description: "D-Flash KV$ Injection Algorithm"
category: "AI"
tags: ["AI"]
cover: "/assets/images/dflash_kv_cache_prefill_update.png"
---

# D-Flash, Who are You?
D-Flash is a speculative decoding model that uses a diffusion-like model to generate `K` draft tokens in parallel. This is similar to Medusa technique but instead of just being a simple MLP, D-Flash utilizes the expressive power of the attention-mechanism. The ability to generate tokens in parallel while also being expressive are two of the qualities that have brought D-Flash as the new standard and have been catalyst to new ideas like gemma4-Difusion and D-Spark. 

# D‑Flash, D-Flash, Why so Clever?

While D-Flash's model architecture is certainly clever, there is also a second important innovation: KV cache injection. 

Like EAGLE, D-Flash also conditions its speculations based on a subset of hidden states from the target model. However, the compressed projection of these hidden states are injected directly to the KV$ of the draft model... *mostly* IN PARALLEL!

This is huge. Since the latency of speculations for D-Flash is not tightly bound to number of speculative tokens, the D-Flash model is deeper (has more layers) than EAGLE models (which are typically a single attention layer). This means that when we inject the KV$, we do not have compute the attention KV$ is a stack format, we can compute them all at once (taking advantage of the extra FLOPS we already assume we have for speculative decoding). Let's go into the specifics as to how this is done in vLLM 0.23.0. 

# KV$ Injection


# The Core Insight: Rich Embeddings + Parallel KV Projection

During the Target model’s prefill, D‑Flash extracts hidden states from multiple layers. These hidden states are concatenated and then linearly compressed into a single “rich embedding” per token. This embedding is surprisingly expressive: it contains enough information for the draft model to reconstruct all of its own KV cache entries.

Here’s the clever part:

•  Every draft‑model layer has its own KV projection weights.
•  Instead of applying each projection separately, D‑Flash concatenates all KV projection matrices into one giant block matrix.
•  It then feeds the compressed embedding through this block once.

This produces the entire draft model KV cache in a single matrix multiplication.

That parallelization is the reason D‑Flash’s TTFT is so fast. You avoid running the draft model layer‑by‑layer; instead, you generate all KV entries in one shot.

# maybe use below

If you try to use a draft model directly, TTFT suffers because the model must run a full forward pass over the entire input sequence to populate its KV cache. Every layer must produce its own K/V projections, and you pay that cost layer‑by‑layer.

D‑Flash also needs to update the draft model’s KV cache — but the cost is much smaller. The trick is that re-use computed hidden states from the Target model. 

# After Prefill: Normal KV Updates Resume

Once prefill is done, the draft model behaves normally. Future KV updates come from the queries you feed it during speculative decoding. The “rich embedding → giant KV matmul” trick is only needed during prefill.

# A Subtle but Beautiful Detail

The KV projection weights are trained to accept inputs from two very different sources:

•  regular draft‑model hidden states (during normal decoding)
•  compressed Target‑model embeddings (during prefill)

The fact that these projections learn a shared representation space — one that works for both — is honestly remarkable. It’s one of those “neural networks quietly doing something elegant” moments.