# Transformer Architecture in Practice

The transformer has become the default architecture for nearly all language modeling work since roughly 2018, and has since been adapted successfully to vision, audio, protein folding, and multimodal tasks. This note covers the practical details of training and deploying transformer models.

## Components Review

A standard transformer block contains: multi-head self-attention, a position-wise feedforward network (typically two linear layers with a nonlinearity between them), residual connections around each sub-layer, and layer normalization (either pre-norm or post-norm).

Modern implementations use pre-norm (layer normalization before the sub-layer, with the residual added after). This configuration trains more stably at large scale than the original post-norm design. RMSNorm is a common simplification of LayerNorm that drops the centering term — it is roughly as effective and slightly faster.

## Attention Patterns

Vanilla self-attention has quadratic cost in sequence length, which becomes prohibitive for long contexts. Many approximations exist:

- **Sparse attention** restricts each token to attend to a subset (sliding window, strided, block-sparse).
- **Linear attention** replaces softmax with a kernel that admits O(n) computation but often underperforms.
- **Flash attention** doesn't change the computation; it reorders memory access to minimize HBM traffic and is now standard in production.
- **Rotary position embeddings (RoPE)** encode position multiplicatively within the attention computation rather than additively at the input.

## Scaling

Doubling parameter count roughly halves loss in a predictable log-linear relationship up through the Chinchilla-optimal point; beyond that, doubling tokens is more efficient per FLOP. Most frontier open-weight models as of 2024 are trained on 10–15 trillion tokens.

Mixture-of-experts (MoE) increases parameter count without proportional compute increase. Only a fraction of experts activate per token; inference is cheaper than a dense model of equivalent parameter count.

## Training Stack

Modern training typically uses: AdamW optimizer with β₁=0.9, β₂=0.95, weight decay 0.1, and a cosine learning rate schedule with warmup. FSDP or ZeRO-3 is standard for model parallelism. Gradient checkpointing trades memory for compute on the backward pass. Mixed-precision training in bf16 (not fp16; bf16 has a wider dynamic range) is now universal.

## Inference Optimizations

Inference servers use KV-caching to avoid recomputing attention for previously-generated tokens. Speculative decoding pairs a small draft model with the target model to generate multiple tokens per target-model call. Quantization to int8 or int4 is commonplace; accuracy loss is generally small for 8-bit and manageable for 4-bit with modern quantization schemes (GPTQ, AWQ, exl2).

Batching across concurrent requests (continuous batching) is essential for serving efficiency. vLLM and TensorRT-LLM are common choices. Throughput is often limited by memory bandwidth, not compute, so model weights and the KV cache compete for the same bandwidth budget.

## When It Fails

Transformers struggle with tasks requiring long chains of precise reasoning, explicit state tracking, algorithmic computation, and retrieval from very long contexts. Their sample efficiency remains poor relative to humans. In-context learning is powerful but brittle. Chain-of-thought prompting helps on some tasks at significant token cost; reasoning fine-tuning (o1-style) is the current frontier.
