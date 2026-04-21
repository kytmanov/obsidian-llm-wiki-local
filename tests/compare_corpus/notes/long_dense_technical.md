# Deep Learning Foundations

## Overview

Deep learning is a family of machine learning methods based on artificial neural networks with representation learning. It differs from classical machine learning primarily in how features are acquired: rather than relying on hand-engineered features, deep networks learn a hierarchy of representations directly from data. The resulting models dominate benchmarks in computer vision, natural language processing, speech recognition, and reinforcement learning.

## Neural Network Fundamentals

At the heart of every deep network is the artificial neuron. A neuron takes a weighted sum of its inputs, adds a bias, and applies a nonlinear activation function. Mathematically: y = σ(Wx + b), where W is the weight matrix, x is the input vector, b is the bias, and σ is the activation.

Common activations include the sigmoid (1 / (1 + e⁻ˣ)), hyperbolic tangent, ReLU (max(0, x)), Leaky ReLU, GELU, and Swish. ReLU became dominant after 2012 because it avoids the vanishing gradient problem that plagued deep sigmoid networks. GELU is now standard in transformer architectures because it offers smoother gradients and matches empirical results better than ReLU on many language tasks.

A feedforward neural network stacks neurons into layers. Each layer is a linear transformation followed by an activation. Depth — the number of stacked layers — is what makes a network "deep." Early work established that a sufficiently wide single hidden layer can approximate any continuous function (the universal approximation theorem), but depth provides exponential compactness for many practical functions.

## Backpropagation

Training a neural network means adjusting its weights to minimize a loss function. Backpropagation is the algorithm that computes gradients of the loss with respect to every weight, by applying the chain rule of calculus layer by layer from output back to input. Without backpropagation, training deep networks at any reasonable scale would be computationally impossible.

Gradient descent then uses these gradients to update the weights. The vanilla update rule is w ← w - η∇L(w), where η is the learning rate. In practice, modern optimizers like Adam, AdamW, and RMSProp maintain running estimates of gradient first and second moments for per-parameter adaptive learning rates.

## Convolutional Neural Networks

Convolutional neural networks (CNNs) dominated computer vision from roughly 2012 (the AlexNet breakthrough on ImageNet) through the late 2010s, and remain dominant in many vision applications. A convolutional layer slides a small weight kernel (e.g., 3×3) across the input, computing a dot product at each position. This gives two critical properties: parameter sharing (the same kernel is applied everywhere) and translation equivariance (shifting the input shifts the output accordingly).

Pooling layers downsample feature maps. Max pooling takes the maximum over a local window; average pooling takes the mean. Modern architectures often skip pooling in favor of strided convolutions.

ResNet introduced skip connections that allow gradients to flow directly across many layers, enabling very deep networks (50, 101, 152 layers) to be trained effectively. This residual learning idea later proved essential to transformers as well.

## Recurrent Neural Networks

RNNs process sequential data by maintaining a hidden state that is updated at each timestep based on both the current input and the previous hidden state: hₜ = σ(W_h hₜ₋₁ + W_x xₜ + b). Vanilla RNNs suffer from the vanishing and exploding gradient problems, making them hard to train on long sequences.

LSTMs (Long Short-Term Memory) solved this by introducing gated cells that can selectively remember or forget information. GRUs (Gated Recurrent Units) simplified the LSTM while retaining most of its capability. Both architectures dominated NLP for several years before being displaced by transformers.

## Attention and Transformers

The attention mechanism lets a model weight different parts of its input dynamically. Given queries Q, keys K, and values V, scaled dot-product attention computes softmax(QKᵀ / √d_k) V. Multi-head attention runs several attention computations in parallel and concatenates them.

The transformer architecture, introduced in "Attention Is All You Need" (2017), is built entirely from self-attention and feedforward layers, with no recurrence. Transformers parallelize well on GPUs, scale to very large datasets, and power virtually every modern large language model.

Positional encoding injects order information because attention is permutation-invariant by default. Sinusoidal encodings, learned embeddings, rotary position embeddings (RoPE), and ALiBi are all in use.

## Training Dynamics

Batch normalization normalizes activations within a mini-batch, stabilizing training and acting as a weak regularizer. Layer normalization, which normalizes across features for each sample independently, is preferred in transformers. Both methods dramatically accelerate convergence.

Regularization techniques include L1/L2 weight decay, dropout (randomly zeroing activations during training), data augmentation, and early stopping. Dropout is somewhat out of favor in very large language models; weight decay and careful data curation do most of the regularization work.

The choice of loss function matters: cross-entropy for classification, mean squared error for regression, contrastive losses for representation learning, and composite losses for multi-task setups.

## Scaling Laws

Empirical scaling laws (Kaplan et al., Hoffmann et al.) relate model quality to three quantities: parameter count, training data size, and compute budget. Roughly, loss decreases as a power law in each, and the three should scale in rough proportion for compute-optimal training. The Chinchilla result showed that earlier large models were undertrained and that doubling tokens while keeping parameters fixed often helps more than the reverse.

## Representation Learning

Deep networks learn hierarchical representations: early layers detect simple features (edges, textures), middle layers compose these into parts (eyes, wheels), and later layers represent objects or concepts. This hierarchy is not imposed architecturally; it emerges from training.

Self-supervised learning has become dominant. Masked language modeling (BERT), autoregressive language modeling (GPT), contrastive learning (SimCLR, CLIP), and masked image modeling (MAE) all learn without labels.

## Generative Models

Generative models learn the data distribution. Variational autoencoders (VAEs), generative adversarial networks (GANs), normalizing flows, autoregressive models, and diffusion models each offer different trade-offs between sample quality, diversity, training stability, and inference speed. Diffusion models currently dominate high-quality image generation; autoregressive transformers dominate text.

## Interpretability

Understanding what deep networks actually compute is an open research problem. Feature visualization, saliency maps, probing classifiers, circuit analysis, and sparse autoencoders are all attempts to open the black box. Mechanistic interpretability has produced partial accounts of how small transformers perform specific tasks, but a complete picture of a frontier language model remains out of reach.

## Open Problems

Despite spectacular progress, deep learning faces significant open questions: sample efficiency remains poor relative to human learning; out-of-distribution generalization is unreliable; reasoning over long chains of inference is brittle; energy costs for training frontier models are enormous; and alignment of increasingly capable systems with human values is an active research frontier.
