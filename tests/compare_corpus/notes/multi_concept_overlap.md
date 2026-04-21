# Mathematical Foundations of Neural Networks

Neural networks sit at the intersection of three mathematical fields: linear algebra, multivariable calculus, and probability theory.

## Linear Algebra

Every layer of a neural network is fundamentally a linear transformation followed by a nonlinearity. The weight matrix W maps input vector x to a new vector Wx. Understanding matrix multiplication, rank, eigenvalues, and singular value decomposition is essential for analyzing network behavior. Batch matrix multiplication on GPUs is the dominant compute cost in modern training.

The dot product Wx expresses a projection of the input into a new coordinate system. When W has full rank, information is preserved; when it is low-rank, the layer performs compression. LoRA and other low-rank adaptation techniques exploit this explicitly, factorizing weight updates as W + BA where B and A are thin rectangular matrices.

## Multivariable Calculus

Gradient descent is an exercise in partial derivatives. The loss L(w) is a scalar function of millions of parameters; training requires computing ∂L/∂wᵢ for every wᵢ. The chain rule, applied recursively, is what backpropagation implements.

Higher-order derivatives matter too. Hessians inform second-order optimization methods like natural gradient descent and K-FAC. Jacobians describe how the network output changes with input — a central quantity in adversarial robustness and sensitivity analysis.

The Jacobian of a single layer is straightforward: it is the weight matrix itself (for the linear part) multiplied by the diagonal matrix of activation derivatives. Stacking these across layers gives the end-to-end Jacobian whose properties determine gradient flow stability.

## Probability Theory

Neural networks are often cast as probabilistic models. Classification networks output a categorical distribution over classes; regression networks can output parameters of a Gaussian. The loss functions we use (cross-entropy, MSE) correspond to maximum likelihood estimation under different likelihood assumptions.

Bayesian deep learning treats weights as distributions rather than point estimates, offering principled uncertainty estimates. Variational inference approximates the posterior, and methods like MC dropout, deep ensembles, and Laplace approximation all fall under this umbrella.

## Where They Meet

These three fields are inseparable in practice. Backpropagation is linear algebra driving calculus. Optimization landscapes are probability distributions over loss surfaces. Attention mechanisms are weighted averages — linear algebra — normalized by softmax — a probability distribution.

The unifying lesson is that neural networks, despite their apparent complexity, rest on mathematical foundations centuries old. Progress in understanding them often comes from applying old mathematical tools in new ways.
