# [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572)

**Source Code:**

**Datasets:** Synthetic (points on the unit circle, $n_0 = 2$), MNIST

**Author:** Arthur Jacot, Franck Gabriel, Clément Hongler (EPFL, Imperial College London)

**Journal:** NeurIPS

**Year of Submission:** 2018 (v4 in Feb 2020)

**Youtube:**

## What problem does it solve?

- We already knew a network **at initialization** is a Gaussian process in the infinite-width limit. Nobody knew what happens **during training**, for arbitrary depth.
- The parameter-space cost $C \circ F^{(L)}$ is highly non-convex (saddle points, many local minima), so we cannot say what gradient descent converges to.
- Why do heavily over-parametrized networks generalize, and why does early stopping help?

The move the paper makes: stop watching the parameters $\theta$, watch the function $f_\theta$ itself. In function space the cost **is** convex. All the non-convexity gets pushed into a single object — the kernel that maps "the change the cost wants" into "the change the network actually makes". That object is the Neural Tangent Kernel.

## How does it solve it?

### Setup: NTK parametrization

Same network as usual, but the $\frac{1}{\sqrt{n_\ell}}$ is pulled *out of* the initialization and put *into* the forward pass, and every parameter is initialized as plain $\mathcal{N}(0,1)$.

$$\alpha^{(0)}(x) = x$$

$$\tilde{\alpha}^{(\ell+1)}(x) = \frac{1}{\sqrt{n_\ell}} W^{(\ell)} \alpha^{(\ell)}(x) + \beta b^{(\ell)}$$

$$\alpha^{(\ell+1)}(x) = \sigma\left(\tilde{\alpha}^{(\ell+1)}(x)\right), \qquad f_\theta(x) = \tilde{\alpha}^{(L)}(x)$$

where:

- $\tilde{\alpha}^{(\ell)}$: the **preactivations** of layer $\ell$ — the linear part, before the nonlinearity.
- $\alpha^{(\ell)}$: the **activations** of layer $\ell$, i.e. $\sigma$ applied entrywise to the preactivations.
- $n_\ell$: width of layer $\ell$. $n_0$ is the input dimension, $n_L$ the output dimension. The limit taken is $n_1, \dots, n_{L-1} \to \infty$ — input and output stay finite.
- $W^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell+1}}$ and $b^{(\ell)} \in \mathbb{R}^{n_{\ell+1}}$: the connection matrix and bias, every entry drawn iid $\mathcal{N}(0,1)$. Not $\mathcal{N}(0, 1/n_\ell)$ — that variance has been moved into the forward pass instead.
- $\beta$: the bias scale, set to $0.1$. It is a knob, not part of the architecture.
- $\theta$: all parameters flattened into one vector of dimension $P = \sum_{\ell=0}^{L-1}(n_\ell + 1)\,n_{\ell+1}$.
- $f_\theta$: the network function. Note $f_\theta = \tilde{\alpha}^{(L)}$, the last **pre**activation — no nonlinearity on the output.
- $F^{(L)}: \mathbb{R}^P \to \mathcal{F}$, $\theta \mapsto f_\theta$: the **realization function**, the map from parameters to the function they realize. Almost everything below is a statement about its derivative.

```python
def network_fn(x, weights, biases):
    activation = x
    for layer in range(num_layers):                            # layers 0 .. L-1
        preactivation = (weights[layer] @ activation) / sqrt(layer_width[layer]) \
                        + bias_scale * biases[layer]
        if layer == num_layers - 1:
            return preactivation                               # output, no final nonlinearity
        activation = nonlinearity(preactivation)

# weights[layer], biases[layer] ~ N(0, 1)      -- plain unit-variance init
# bias_scale                    = beta = 0.1   -- the paper's bias knob
```

- The set of representable functions $F^{(L)}(\mathbb{R}^P)$ is **identical** to LeCun init. What changes is the **derivative** of the realization function:

$$\partial_{W^{(\ell)}_{ij}} F^{(L)} \ \text{is scaled by}\ \frac{1}{\sqrt{n_\ell}}, \qquad \partial_{b^{(\ell)}_{j}} F^{(L)} \ \text{is scaled by}\ \beta$$

- Rescaling the *gradients* — not the function class — is exactly what makes the $n \to \infty$ limit well behaved.
- Side effect: at large width the connection weights barely influence training; $\beta$ (they use $0.1$) rebalances bias vs weights. Learning rate $1.0$, which behaves like a classical width-100 net at lr $0.01$.
- $\sigma$ assumed Lipschitz, twice differentiable, with bounded second derivative (needed for Theorem 2). ReLU is used in every experiment anyway.

### Training flow (as the paper sees it)

1. Parameters move by ordinary gradient descent (in continuous time, i.e. gradient flow): $\partial_t \theta_p(t) = -\partial_{\theta_p}\left(C \circ F^{(L)}\right)(\theta(t))$.
2. Chain rule ⇒ the *function* moves as $\partial_t f_{\theta(t)} = -\nabla_{\Theta^{(L)}} C\big|_{f_{\theta(t)}}$, i.e. **kernel gradient descent** in function space.
3. The kernel doing the mapping is $\Theta^{(L)}(\theta) = \sum_{p=1}^{P} \partial_{\theta_p} F^{(L)}(\theta) \otimes \partial_{\theta_p} F^{(L)}(\theta)$ — a sum of $P$ outer products of "how the function changes when parameter $p$ wiggles".
4. For a finite network $\Theta^{(L)}$ is random at init and drifts during training, so nothing can be concluded.
5. **Theorem 1:** as $n_1, \dots, n_{L-1} \to \infty$, the NTK at initialization converges *in probability* to a deterministic kernel $\Theta^{(L)}_\infty \otimes \mathrm{Id}_{n_L}$, given in closed form by depth, $\sigma$, init variance and $\beta$. No randomness left.
6. **Theorem 2:** in the same limit $\Theta^{(L)}(t) \to \Theta^{(L)}_\infty \otimes \mathrm{Id}_{n_L}$ **uniformly for $t \in [0,T]$** — the kernel stays put for the whole of training.
7. Therefore an infinitely wide net trained by gradient descent *is* kernel gradient descent with a fixed known kernel. Training becomes a linear ODE in function space.
8. Convergence to a global optimum then reduces to one question — is $\Theta^{(L)}_\infty$ positive definite? (Proposition 2: yes, for $L \ge 2$, non-polynomial Lipschitz $\sigma$, data on the sphere.)

### Inference

Nothing changes at inference — this is an analysis paper, not a method. But the limit gives a closed form for the converged network under least squares:

$$f_{\infty,k}(x) = \underbrace{\kappa_{x,k}^{T}\, \tilde{K}^{-1} y^{*}}_{\text{kernel ridge regression mean}} + \underbrace{\left(f_0(x) - \kappa_{x,k}^{T}\, \tilde{K}^{-1} y_0\right)}_{\text{init fluctuation, zero on the training set}}$$

where:

- $\kappa_{x,k} = \left(K_{kk'}(x, x_i)\right)_{i,k'}$: the vector of kernel similarities between the query $x$ and every training point.
- $\tilde{K} = \left(K_{kk'}(x_i,x_j)\right)_{ik,jk'}$: the $Nn_L \times Nn_L$ Gram matrix of the kernel on the training set. Invertible **iff** the kernel is positive definite on the data — this is where Proposition 2 is needed.
- $y^{*} = \left(f^{*}_k(x_i)\right)_{i,k}$: the training targets.
- $y_0 = \left(f_{0,k}(x_i)\right)_{i,k}$: the *random* predictions the network made at initialization, before any training.
- $f_0(x)$: the network at initialization evaluated at the query point.

Reading it: the first term is the MAP estimate under a Gaussian prior $f_k \sim \mathcal{N}(0, \Theta^{(L)}_\infty)$, equivalently kernel ridge regression as the regularization $\lambda \to 0$. The second term is a centered Gaussian whose variance vanishes exactly on the training points — the network never fully forgets its initialization, but the leftover is invisible on the training data.

```python
similarity_to_train = [limit_ntk(x, train_x) for train_x in train_inputs]
train_gram          = [[limit_ntk(a, b) for b in train_inputs] for a in train_inputs]
init_preds_on_train = [network_at_init(train_x) for train_x in train_inputs]

kernel_ridge_mean = similarity_to_train.T @ inv(train_gram) @ train_targets
init_fluctuation  = network_at_init(x) \
                    - similarity_to_train.T @ inv(train_gram) @ init_preds_on_train

converged_prediction = kernel_ridge_mean + init_fluctuation
```

### Equations

#### 1. Function space and its two inner products

The data distribution $p^{in}$ is the empirical distribution on the training set, $p^{in} = \frac{1}{N}\sum_{i=1}^{N}\delta_{x_i}$. Function space is $\mathcal{F} = \{f : \mathbb{R}^{n_0} \to \mathbb{R}^{n_L}\}$, carrying a **data seminorm**:

$$\langle f, g \rangle_{p^{in}} = \mathbb{E}_{x \sim p^{in}}\left[f(x)^{T} g(x)\right]$$

and, for any kernel $K$, a **kernel inner product**:

$$\langle f, g \rangle_{K} = \mathbb{E}_{x, x' \sim p^{in}}\left[f(x)^{T} K(x,x') g(x')\right]$$

where:

- $K: \mathbb{R}^{n_0} \times \mathbb{R}^{n_0} \to \mathbb{R}^{n_L \times n_L}$, with $K(x,x') = K(x',x)^{T}$ — a matrix-valued (multi-dimensional) kernel, because the network has $n_L$ outputs.
- $\langle \cdot,\cdot\rangle_{p^{in}}$ is only a **semi**norm: two functions agreeing on the $N$ training points are indistinguishable under it. This is precisely why the kernel matters — it is what decides the rest.
- $K$ is *positive definite w.r.t.* $\|\cdot\|_{p^{in}}$ if $\|f\|_{p^{in}} > 0 \implies \|f\|_{K} > 0$: no nonzero-on-the-data function is invisible to the kernel.

#### 2. Kernel gradient — how a derivative on the data becomes a change everywhere

The cost $C$ only sees $f$ at the $N$ data points, so its functional derivative $\partial^{in}_f C |_{f_0}$ is a linear form, representable as $\langle d|_{f_0}, \cdot \rangle_{p^{in}}$ for some $d|_{f_0} \in \mathcal{F}$. The kernel turns that into a function defined on **all** of $\mathbb{R}^{n_0}$:

$$\nabla_K C\big|_{f_0}(x) = \frac{1}{N} \sum_{j=1}^{N} K(x, x_j)\, d|_{f_0}(x_j)$$

where:

- $d|_{f_0}$: the dual element of the cost derivative. For least squares it is simply $d = f_0 - f^{*}$, the residual on the training points.
- $K(x, x_j)$: how strongly a change at training point $x_j$ propagates to the query point $x$. **This factor is the entire generalization story** — $d$ lives only on the data, and $K$ is what smears it outward.
- $\frac{1}{N}\sum_j$: the expectation over $p^{in}$, written out.

Kernel gradient descent is then the ODE

$$\partial_t f(t) = -\nabla_K C\big|_{f(t)}$$

and along it the cost decreases at a rate given by the kernel norm of the residual:

$$\partial_t C\big|_{f(t)} = -\left\langle d|_{f(t)},\, \nabla_K C|_{f(t)}\right\rangle_{p^{in}} = -\left\lVert d|_{f(t)} \right\rVert_{K}^{2}$$

Reading it: the cost is non-increasing always, and **strictly** decreasing whenever $\|d\|_{p^{in}} > 0$ — provided $K$ is positive definite. Convex plus bounded below plus PD kernel ⇒ global minimum. That single implication is why the rest of the paper is about positive definiteness.

```python
def kernel_gradient(x, kernel, cost_derivative):
    return mean([kernel(x, train_x) @ cost_derivative(train_x)
                 for train_x in train_inputs])
```

#### 3. The NTK itself

$$\Theta^{(L)}(\theta) = \sum_{p=1}^{P} \partial_{\theta_p} F^{(L)}(\theta) \otimes \partial_{\theta_p} F^{(L)}(\theta)$$

where:

- $\partial_{\theta_p} F^{(L)}(\theta) \in \mathcal{F}$: the function-space direction that parameter $p$ can push in — "if I nudge $\theta_p$, how does the whole function change?".
- $\otimes$: outer product, so each term is a rank-one kernel; the sum over all $P$ parameters is PSD by construction.
- The dependence on $\theta$ is the problem: unlike a fixed kernel, $\Theta^{(L)}$ is random at initialization and moves as $\theta$ moves. Theorems 1 and 2 kill both objections in the infinite-width limit.

With this, the parameter-space gradient descent of step 1 in the training flow becomes exactly

$$\partial_t f_{\theta(t)} = -\nabla_{\Theta^{(L)}(\theta(t))} C \big|_{f_{\theta(t)}}$$

i.e. gradient descent on parameters **is** kernel gradient descent on functions — for any network, any width, no limit needed yet.

#### 4. Warm-up: random features (why the theorems are believable)

Draw $P$ random functions $f^{(p)}$ whose non-centered covariance is the kernel you want, $\mathbb{E}\left[f^{(p)}_k(x) f^{(p)}_{k'}(x')\right] = K_{kk'}(x,x')$, and build a **linear** model out of them:

$$f^{lin}_\theta = \frac{1}{\sqrt{P}} \sum_{p=1}^{P} \theta_p f^{(p)} \qquad\Longrightarrow\qquad \partial_{\theta_p} F^{lin}(\theta) = \frac{1}{\sqrt{P}} f^{(p)}$$

Its tangent kernel is then, by the same definition as in part 3,

$$\tilde{K} = \sum_{p=1}^{P} \partial_{\theta_p} F^{lin} \otimes \partial_{\theta_p} F^{lin} = \frac{1}{P} \sum_{p=1}^{P} f^{(p)} \otimes f^{(p)} \ \xrightarrow[P \to \infty]{} \ K$$

where:

- The $\frac{1}{\sqrt{P}}$ in the model becomes $\frac{1}{P}$ in the kernel — which is exactly a sample average, so the law of large numbers applies.
- Because $F^{lin}$ is linear, $\partial_{\theta_p} F^{lin}$ does **not** depend on $\theta$: the kernel is automatically constant during training. Nothing to prove.
- The whole content of Theorems 1 and 2 is that a real, nonlinear ANN behaves like this anyway — with $\frac{1}{\sqrt{n_\ell}}$ playing the role of $\frac{1}{\sqrt{P}}$.

```python
# E[basis_fn(x) * basis_fn(x2)] == target_kernel(x, x2) for each random basis function
def linear_model(x, coefficients):
    return sum(coefficient * basis_fn(x)
               for coefficient, basis_fn in zip(coefficients, random_basis)) / sqrt(num_features)

tangent_kernel = mean([outer(basis_fn, basis_fn) for basis_fn in random_basis])
```

#### 5. Recursion A — the Gaussian process covariance (Proposition 1)

At initialization, in the infinite-width limit, the $n_L$ output functions tend to iid centered Gaussian processes of covariance $\Sigma^{(L)}$, defined recursively:

$$\Sigma^{(1)}(x,x') = \frac{1}{n_0} x^{T} x' + \beta^{2}$$

$$\Sigma^{(\ell+1)}(x,x') = \mathbb{E}_{f \sim \mathcal{N}\left(0, \Sigma^{(\ell)}\right)}\left[\sigma(f(x))\, \sigma(f(x'))\right] + \beta^{2}$$

where:

- $\Sigma^{(\ell)}(x,x')$: the covariance between the layer-$\ell$ preactivations at inputs $x$ and $x'$, taken over the random initialization.
- $\frac{1}{n_0}x^{T}x'$: the base case is just the (normalized) input dot product — layer 1 is linear in the input.
- $\beta^{2}$: the bias contributes an additive constant at every layer, since $b^{(\ell)} \sim \mathcal{N}(0,1)$ scaled by $\beta$.
- The expectation is over a *2-dimensional* Gaussian: only the joint law of $\left(f(x), f(x')\right)$ matters, so this is a 2D integral, computable in closed form for ReLU (arc-cosine kernels).
- Why a Gaussian appears at all: each preactivation is $\frac{1}{\sqrt{n}}\sum_j W_{ij}\alpha_j$, a CLT-normalized sum of $n$ terms.

#### 6. Recursion B — the Neural Tangent Kernel (Theorem 1)

First the same expectation but through the **derivative** of the nonlinearity:

$$\dot{\Sigma}^{(\ell+1)}(x,x') = \mathbb{E}_{f \sim \mathcal{N}\left(0, \Sigma^{(\ell)}\right)}\left[\dot{\sigma}(f(x))\, \dot{\sigma}(f(x'))\right]$$

then the kernel itself:

$$\Theta^{(1)}_{\infty}(x,x') = \Sigma^{(1)}(x,x')$$

$$\Theta^{(\ell+1)}_{\infty}(x,x') = \underbrace{\Theta^{(\ell)}_{\infty}(x,x')\, \dot{\Sigma}^{(\ell+1)}(x,x')}_{\text{learning done by all lower layers}} + \underbrace{\Sigma^{(\ell+1)}(x,x')}_{\text{learning done by the last layer}}$$

where:

- $\dot{\sigma}$: derivative of the nonlinearity (defined a.e. by Rademacher; for ReLU it is the step function, so $\dot{\Sigma}$ measures how often two inputs land on the same side of a hidden unit).
- **Second summand** $\Sigma^{(\ell+1)}$: the last layer's weights multiply the previous activations directly, so their contribution to the tangent kernel is just the covariance of those activations — plain linear regression on fixed features.
- **First summand** $\Theta^{(\ell)}_{\infty} \dot{\Sigma}^{(\ell+1)}$: everything below, carried up by the chain rule. Each extra layer multiplies the accumulated kernel by one more derivative factor $\dot{\Sigma}$. This is backprop written as a recursion on kernels.
- $\Theta^{(L)}_{\infty}$ depends on: depth $L$, nonlinearity $\sigma$, init variance, $\beta$. It does **not** depend on the width, the dataset, or the random draw. Architecture choice = kernel choice.

Transcribed one-to-one from the two equations above — readable, but **not** how you would actually compute it (see part 7):

```python
def activation_cov(depth, x, y):
    """Sigma^(depth)(x, y) -- schematic."""
    if depth == 1:
        return dot(x, y) / input_dim + bias_scale ** 2
    incoming = activation_cov(depth - 1, x, y)
    return expect_over_gaussian(incoming,
                                lambda fx, fy: nonlinearity(fx) * nonlinearity(fy)) \
           + bias_scale ** 2


def activation_cov_derivative(depth, x, y):
    """Sigma_dot^(depth)(x, y) -- same expectation, through the derivative."""
    incoming = activation_cov(depth - 1, x, y)
    return expect_over_gaussian(incoming,
                                lambda fx, fy: nonlinearity_grad(fx) * nonlinearity_grad(fy))


def limit_ntk(depth, x, y):
    """Theta_inf^(depth)(x, y) -- schematic."""
    if depth == 1:
        return activation_cov(1, x, y)
    lower_layer_learning = limit_ntk(depth - 1, x, y) * activation_cov_derivative(depth, x, y)
    last_layer_learning  = activation_cov(depth, x, y)
    return lower_layer_learning + last_layer_learning
```

#### 7. Pseudocode: actually computing $\Theta^{(L)}_{\infty}$

The schematic version hides a real subtlety. `expect_over_gaussian(incoming, ...)` cannot be evaluated from the single scalar $\Sigma^{(\ell)}(x,y)$: the expectation is over the *joint* law of the pair $\left(f(x), f(y)\right)$, and that 2D Gaussian is specified by **three** numbers, not one —

$$\Sigma^{(\ell)}(x,x), \qquad \Sigma^{(\ell)}(x,y), \qquad \Sigma^{(\ell)}(y,y)$$

the two diagonal entries setting the scale and the off-diagonal setting the correlation. So the real algorithm carries three running scalars and sweeps layers bottom-up instead of recursing:

```python
def limit_ntk(x, y, depth, bias_scale):
    """Theta_inf^(depth)(x, y) for a fully connected net -- Theorem 1.

    Carries the full 2x2 covariance of (f(x), f(y)) layer by layer, because
    every Gaussian expectation below needs all three of its entries.
    """
    input_dim = len(x)

    # ---- layer 1: Sigma^(1) is the normalised input Gram plus the bias ----
    cov_xx = dot(x, x) / input_dim + bias_scale ** 2
    cov_yy = dot(y, y) / input_dim + bias_scale ** 2
    cov_xy = dot(x, y) / input_dim + bias_scale ** 2

    ntk = cov_xy                                        # Theta^(1) = Sigma^(1)

    for layer in range(2, depth + 1):
        # Sigma^(layer) and Sigma_dot^(layer), under f ~ N(0, [[xx, xy], [xy, yy]])
        next_cov_xy, derivative_cov = gaussian_expectations(cov_xx, cov_yy, cov_xy)

        # the diagonals propagate too -- each is its own 1-point expectation
        next_cov_xx, _ = gaussian_expectations(cov_xx, cov_xx, cov_xx)
        next_cov_yy, _ = gaussian_expectations(cov_yy, cov_yy, cov_yy)

        next_cov_xy += bias_scale ** 2
        next_cov_xx += bias_scale ** 2
        next_cov_yy += bias_scale ** 2

        # Theorem 1: lower layers carried up by one derivative factor,
        #            plus this layer's own last-layer contribution
        lower_layer_learning = ntk * derivative_cov
        last_layer_learning  = next_cov_xy
        ntk = lower_layer_learning + last_layer_learning

        cov_xx, cov_yy, cov_xy = next_cov_xx, next_cov_yy, next_cov_xy

    return ntk
```

The one primitive left is the pair of 2D Gaussian expectations. For ReLU both have closed forms (the arc-cosine kernels of Cho & Saul — used by the paper's experiments, but not derived in it):

```python
def gaussian_expectations(cov_xx, cov_yy, cov_xy):
    """Returns (E[sigma(fx) sigma(fy)], E[sigma'(fx) sigma'(fy)]) for ReLU."""
    scale  = sqrt(cov_xx * cov_yy)
    cosine = clip(cov_xy / scale, -1.0, 1.0)           # correlation of fx, fy
    angle  = arccos(cosine)                            # 0 = identical, pi = opposite

    activation_cov = scale * (sin(angle) + (pi - angle) * cosine) / (2 * pi)
    derivative_cov = (pi - angle) / (2 * pi)           # P(fx > 0 and fy > 0)
    return activation_cov, derivative_cov
```

For an arbitrary nonlinearity there is no closed form, so estimate the same two numbers numerically:

```python
def gaussian_expectations(cov_xx, cov_yy, cov_xy, num_samples=100_000):
    """Same two expectations for any sigma, by Monte Carlo (or Gauss-Hermite)."""
    covariance = [[cov_xx, cov_xy],
                  [cov_xy, cov_yy]]
    fx, fy = sample_multivariate_normal(mean=[0, 0], cov=covariance, size=num_samples).T
    return (mean(nonlinearity(fx) * nonlinearity(fy)),
            mean(nonlinearity_grad(fx) * nonlinearity_grad(fy)))
```

Reading the loop:

- `ntk` accumulates $\Theta^{(\ell)}_{\infty}$; at every layer it is multiplied by `derivative_cov` ($\dot{\Sigma}^{(\ell)}$) and has `next_cov_xy` ($\Sigma^{(\ell)}$) added. That single line **is** Theorem 1.
- `derivative_cov` for ReLU is the probability that both inputs activate the same hidden unit. Inputs that always fire together keep their kernel mass; inputs that disagree lose it at every layer — that is how depth shapes the kernel.
- The width $n$ never appears. Neither does the random draw. The whole point of the limit is that the kernel is a deterministic function of $(x, y, L, \sigma, \beta)$.
- Cost is $O(L)$ per pair, so building the $N \times N$ Gram matrix needed for the closed-form solution of part 9 is $O(N^{2}L)$ — the usual kernel-method quadratic wall.
- For the whole dataset at once, replace the three scalars with three $N \times N$ matrices and apply `gaussian_expectations` entrywise; the paper's experiments compute $\Theta^{(4)}_{\infty}$ this way (approximated by an $n = 10000$ network rather than the closed form).

#### 8. Why the kernel can be constant while the network still learns (Remark 4)

Each individual hidden activation moves by $O\!\left(\frac{1}{\sqrt{n}}\right)$ during training — vanishing. But there are $n$ of them, so their **collective** movement is $O(1)$. Lower layers do learn, yet no single unit moves enough to perturb the kernel. This is the whole miracle, and also the whole limitation (see rebuttals).

#### 9. Least squares: the linear ODE

The cost and the resulting dynamics:

$$C(f) = \frac{1}{2}\lVert f - f^{*} \rVert^{2}_{p^{in}} \qquad\Longrightarrow\qquad \partial_t f_t = \Phi_K\left(\langle f^{*} - f_t, \cdot \rangle_{p^{in}}\right)$$

Because the residual enters linearly, this is a **linear** ODE, solved by an operator exponential:

$$f_t = f^{*} + e^{-t\Pi}\left(f_0 - f^{*}\right), \qquad e^{-t\Pi} = \sum_{k=0}^{\infty}\frac{(-t)^{k}}{k!}\Pi^{k}$$

where the operator $\Pi$ is the kernel restricted to the data:

$$\Pi(f)_k(x) = \frac{1}{N}\sum_{i=1}^{N}\sum_{k'=1}^{n_L} f_{k'}(x_i)\, K_{kk'}(x_i, x)$$

where:

- $\Phi_K$: the map that turns a linear form on $\mathcal{F}$ into a function, via the kernel — the same operation as in part 2.
- $\Pi$: has at most $Nn_L$ positive eigenvalues; its eigenfunctions $f^{(1)}, \dots, f^{(Nn_L)}$ are exactly the **kernel principal components** of the data w.r.t. $K$, and the eigenvalue $\lambda_i$ is the variance captured by component $i$.
- $e^{-t\Pi}$ shares those eigenfunctions with eigenvalues $e^{-t\lambda_i}$ — which is the whole reason for diagonalizing.

Decompose the initial error along the eigenspaces, $f^{*} - f_0 = \Delta^{0}_{f} + \Delta^{1}_{f} + \dots + \Delta^{Nn_L}_{f}$:

$$f_t = f^{*} + \Delta^{0}_{f} + \sum_{i=1}^{Nn_L} e^{-t\lambda_i}\, \Delta^{i}_{f}$$

where:

- $\Delta^{i}_{f} \propto f^{(i)}$: the part of the initial error along the $i$-th kernel principal component. It decays at its **own** exponential rate $e^{-t\lambda_i}$ — big-$\lambda$ directions are fit fast, small-$\lambda$ directions slowly.
- $\Delta^{0}_{f}$: the part in the null space of $\Pi$. It has $\lambda = 0$, so it **never** decays — the component the dynamics simply cannot reach.
- **Theoretical motivation for early stopping:** stopping at finite $t$ means the large-$\lambda$ components have converged and the small-$\lambda$ ones have not. For kernels like the RBF, small $\lambda$ means high frequency, i.e. the noisy directions. Early stopping is therefore spectral filtering, not a heuristic.

```python
def kernel_operator(fn, x):                       # Pi(f) evaluated at x
    return mean([kernel(train_x, x) @ fn(train_x) for train_x in train_inputs])


eigenvalues, principal_components = eig(kernel_operator)     # kernel PCA of the training set
initial_error   = target_fn - init_fn
error_along     = project(initial_error, principal_components)
unreachable     = project_onto_nullspace(initial_error, kernel_operator)

def function_at_time(t, x):
    decayed = sum(exp(-t * eigenvalue) * component(x)
                  for eigenvalue, component in zip(eigenvalues, error_along))
    return target_fn(x) + unreachable(x) + decayed
```

### Model

Fully connected MLPs only. ReLU, all hidden widths equal ($n \in \{50, 100, 500, 1000, 10000\}$), depth $L = 4$ in the experiments, $n_L = 1$, $\beta = 0.1$, learning rate $1.0$.

## How is this paper novel?

- Prior work described infinitely wide nets **at initialization** (the GP limit). This paper describes them **during training**, at arbitrary depth. Shallow dynamics were known; deep was open.
- Introduces $\Theta^{(L)} = \sum_p \partial_{\theta_p} F^{(L)} \otimes \partial_{\theta_p} F^{(L)}$ and shows it is the right object: the entire non-convex parameter geometry collapses into one fixed PSD kernel in the limit.
- Turns "does gradient descent converge?" into "is this kernel positive definite?", then answers it for data on a sphere with non-polynomial $\sigma$ and $L \ge 2$.
- Gives a **derivation** rather than a hunch for early stopping, via the kernel PCA spectrum.
- Establishes: wide net + gradient descent + squared loss ⟶ kernel ridge regression with $\Theta^{(L)}_\infty$ (as $\lambda \to 0$), plus a Gaussian term that vanishes on the data.

## List of experiments

All experiments: fully connected, ReLU, equal hidden widths $n$, $n_L = 1$, $\beta = 0.1$, learning rate $1.0$.

### 1. Convergence of the NTK (Figure 1)

- $L = 4$, $n \in \{500, 10000\}$, inputs on the unit circle in $\mathbb{R}^2$, target $f^{*}(x) = x_1 x_2$, 10 independent inits each.
- Plot $\Theta^{(4)}(x_0, x)$ for fixed $x_0 = (1,0)$ and $x = (\cos\gamma, \sin\gamma)$, at $t = 0$ and after 200 steps of GD.
- Wider ⇒ far lower variance across inits and a smoother kernel. The **mean** kernel is already nearly identical at both widths.
- After training the NTK "inflates" (grows in magnitude); the inflation is much smaller at $n = 10000$. This is the finite-width violation of Theorem 2, shrinking with $n$ exactly as predicted.

### 2. Kernel regression (Figure 2)

- Train on 4 points of the unit circle for 1000 steps, $n \in \{50, 1000\}$, 10 inits each.
- Compare the empirical spread of $f_{\theta(T)}$ against the theoretical $t \to \infty$ Gaussian, whose 10th/50th/90th percentiles come from $\Theta^{(4)}_\infty$ and $\Sigma^{(4)}$ approximated with an $n = 10000$ network.
- The finite-width function distributions match the limiting Gaussian closely — even at $n = 50$.

### 3. Convergence along a principal component (Figure 3, MNIST)

- $N = 512$ digits, $n_0 = 784$. First 3 kernel principal components w.r.t. the $n = 10000$ NTK via power iteration. Eigenvalues $\lambda_1 = 0.0457$, $\lambda_2 = 0.00108$, $\lambda_3 = 0.00078$.
- The PCA is **non-centered**, so component 1 ≈ the constant function — which is why $\lambda_1$ is an order of magnitude above the rest. Components 2 and 3 are the interesting ones.
- Set the target to $f^{*} = f_{\theta(0)} + 0.5 f^{(2)}$, i.e. make the initial error exactly one principal component. Theory then says the trajectory is a straight line in function space, with $\lVert g_t \rVert_{p^{in}} = 0.5\, e^{-\lambda_2 t}$ and $h_t = 0$.
- Split $f_{\theta(t)} - f^{*}$ into $g_t$ (the part along $f^{(2)}$) and $h_t$ (the orthogonal part), for $n \in \{100, 1000, 10000\}$:
  - $\lVert h_t \rVert$, the deviation from the straight line, shrinks as width grows (Figure 3b).
  - $\lVert g_t \rVert$ converges to the theoretical $0.5 e^{-\lambda_2 t}$ curve as width grows (Figure 3c).
- Curiosity: **narrower networks converge faster in step count.** Explained by the NTK inflation from experiment 1 — inflating the kernel by a factor $a$ is the same as multiplying the learning rate by $a$. Step-count comparisons are therefore confounded; wide nets can simply take a larger learning rate because their kernel is stable.
- Footnote worth remembering: with $\beta = 1.0$ instead of $0.1$, the gap between the 1st and 2nd eigenvalues is about ten times bigger, which makes training harder. The bias scale directly deforms the kernel spectrum.

### Ablation Studies

None as such — the width sweep **is** the ablation, since the entire claim is "behaviour → limit as $n$ grows".

### Efficiency analysis

Not applicable.

## Preliminaries

### Kernel gradient vs ordinary gradient

Ordinary gradient descent in function space would move $f$ only at the training points — the cost derivative is a sum of Diracs, $p^{in} = \frac{1}{N}\sum_i \delta_{x_i}$. A kernel gradient smears that update across input space through the factor $K(x, x_j)$. Everything the model does off the training set is decided by the kernel.

### Positive definiteness

$K$ is PD w.r.t. $\lVert \cdot \rVert_{p^{in}}$ if $\lVert f \rVert_{p^{in}} > 0 \implies \lVert f \rVert_{K} > 0$. Since $\partial_t C = -\lVert d \rVert_K^2$, a PD kernel means the cost strictly decreases whenever the residual on the data is nonzero. Without PD there are error directions the dynamics is blind to.

### Kernel PCA

Eigendecomposition of the Gram matrix of $K$ over the dataset. The eigenfunctions $f^{(i)}$ are directions in function space and $\lambda_i$ is the variance captured by that direction. For an RBF kernel, large $\lambda$ ≈ low frequency, small $\lambda$ ≈ high frequency. The NTK inherits this "smooth components first" ordering, which is what makes early stopping a low-pass filter.

### Gaussian process limit (why it happens)

A preactivation

$$\tilde{\alpha}^{(\ell+1)}_i(x) = \frac{1}{\sqrt{n_\ell}}\sum_{j=1}^{n_\ell} W^{(\ell)}_{ij}\, \alpha^{(\ell)}_j(x) + \beta b^{(\ell)}_i$$

is a $\frac{1}{\sqrt{n}}$-scaled sum of $n$ iid-ish terms ⇒ CLT ⇒ Gaussian, with covariance given by the $\Sigma^{(\ell)}$ recursion. The $\frac{1}{\sqrt{n}}$ is precisely the CLT normalization — which is why the NTK parametrization is the "right" one for taking limits.

### The operator exponential $e^{-t\Pi}$

$$e^{-t\Pi} = \sum_{k=0}^{\infty}\frac{(-t)^{k}}{k!}\Pi^{k}$$

If $\Pi$ has eigenfunctions $f^{(i)}$ with eigenvalues $\lambda_i$, then $e^{-t\Pi}$ has the *same* eigenfunctions with eigenvalues $e^{-t\lambda_i}$. It is just "solve a linear ODE by diagonalizing", lifted from vectors to functions.

### Stochastic boundedness

Theorem 2 needs $\int_0^T \lVert d_t \rVert_{p^{in}}\, dt$ to stay stochastically bounded as the width grows — i.e. the total training signal must not blow up. For least squares this is free, because $\lVert f^{*} - f_t \rVert_{p^{in}}$ is strictly decreasing.

## GPU hours

Not reported. Experiments are tiny (MLPs, $N = 512$ MNIST, at most 4000 steps), though $n = 10000$ dense layers are memory-heavy.

## Key takeaways

1. Gradient descent on parameters **is** kernel gradient descent on functions, always. The only questions are which kernel, and whether it moves.
2. Infinite width ⇒ the kernel is deterministic (Thm 1) and frozen (Thm 2) ⇒ the network is a linear model in function space and training has a closed-form solution.
3. $\Theta^{(L)}_\infty$ depends only on depth, nonlinearity, init variance and $\beta$. Architecture choices are kernel choices.
4. Convergence guarantee follows from positive definiteness of $\Theta^{(L)}_\infty$. Proven for $L \ge 2$, non-polynomial $\sigma$, data on a sphere.
5. Early stopping is spectral truncation: large-eigenvalue kernel principal components are fit first, small (noisy) ones last.
6. Wide net + squared loss $\approx$ kernel ridge regression with $\Theta^{(L)}_\infty$, plus a Gaussian term that is zero on the training data.
7. Finite-width networks **do** deviate — the NTK inflates during training — but the deviation shrinks visibly with width, and even $n = 50$ is qualitatively right.

## What I still do not understand?

- The limit proofs take widths to infinity **sequentially** ($n_1 \to \infty$, then $n_2 \to \infty$, ...). How much does the "all widths simultaneously" version change?
- Rate of convergence: Theorem 2 says the kernel converges, but how fast in $n$? The paper does not say.
- Where does the observed NTK inflation come from quantitatively — is it an $O(1/n)$ correction with a computable constant?
- Proposition 2 requires data on a sphere. How badly does PD fail for real, un-normalized data?
- Why is $\Delta^{0}_{f}$ (the null-space part of the error) empirically small on real datasets? Nothing here says it should be.
- What exactly breaks the constant-kernel picture at realistic widths — is there a threshold, or is it continuous degradation?

## Ideas to pursue

- If early stopping = truncating small-$\lambda$ kernel principal components, then the NTK spectrum of a dataset is a measurable "how much of this task is easy" statistic, computable **before** training as an architecture-selection signal.
- $\Theta^{(L)}_\infty$ is a closed-form function of depth and nonlinearity. Search over $\sigma$ to *design* a kernel with a desired spectrum instead of picking ReLU by habit.
- Label noise should land in the small-$\lambda$ components. Check whether projecting onto the NTK spectrum detects mislabeled examples.
- $\beta$ visibly changes the eigenvalue gap — treat the bias scale as a tunable spectral knob rather than a fixed $0.1$.
- Compare the NTK-predicted training curve against the actual one as a diagnostic: the size of the gap measures how much feature learning a given setup is doing.

## Rebuttals

- The frozen-kernel regime is exactly the regime where the network **does not learn features** — the representation is fixed at initialization and only the readout moves. Deep learning's practical advantage over kernels is widely believed to come from feature learning, so NTK explains convergence far better than it explains why deep nets beat kernel methods.
- Fully connected only. No convolutions, attention, batchnorm or weight sharing.
- Requires the NTK parametrization plus the $\beta$ correction; standard training sits at a different (feature-learning) scaling.
- Infinite width and gradient **flow** (continuous time). No SGD noise, no momentum, no finite-learning-rate effects.
- Their own Figure 1 shows the kernel drifting at realistic widths, so every statement is an approximation whose error is not quantified here.

## Similar papers

- Neal (1996), Lee et al. (2018), Matthews et al. (2018), Daniely et al. (2016) — the GP-at-initialization line this paper extends into training.
- Cho & Saul (2009) — arc-cosine kernels; the closed forms that make the $\Sigma$ / $\dot{\Sigma}$ recursions computable for ReLU.
- Belkin, Ma, Mandal (2018) — kernels also fit random labels yet generalize, the observation that makes the NTK connection interesting rather than damning. See also [double-descent](../double-descent/README.md).
- Chizat & Bach — "lazy training": argues the constant-kernel regime is precisely the *uninteresting* one.
