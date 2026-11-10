# [The Geometric Occam's Razor Implicit in Deep Learning](https://arxiv.org/abs/2111.15090)

## Meta

* Journal   - NeurIPS 2021 workshop (OPT2021)
* Year      - 2021
* Author    - Google, DeepMind (Dherin, Munn, Barrett)
* Code      - Not mentioned
* Slides    - 
* One liner - SGD implicitly penalises the geometric complexity of the learned function (arc length, or a Dirichlet-energy proxy), so a trained 1-D ReLU network is near the shortest curve through the data.
* Model     - 3-layer ReLU network (300 units), ResNet-18
* Datasets  - CIFAR-10, a 1-D toy regression set of 10 points
* Baselines - None

## What problem does it solve?

- An over-parameterized network can fit the training data exactly in many different ways
- We do not know well which of these interpolating solutions SGD selects, and why the selected one generalizes
- The classic Occam's razor counts parameters or the size of the hypothesis space. This does not explain why a larger model can be better than a smaller one (the paper mentions double descent as this paradox, and does not analyse it)
- The paper proposes a different measure of simplicity: the geometric complexity of the learned function. It argues that SGD decreases this complexity implicitly

## How does it solve it?

### Setup

1. See the learned function `f` through its graph `gr(f) = {(x, f(x))}`
2. 1-D regression: geometric complexity = arc length of the graph on the smallest interval that contains the data
3. Higher dimensions: geometric complexity = volume of the graph on the feature polytope `X_D` (the smallest-volume polytope that contains the data points)
4. Approximate the volume to first order. The result is a constant plus the Dirichlet energy of `f`
5. Estimate the Dirichlet energy with a sum over the data points (discrete Dirichlet energy)

### Main results

1. **Start: implicit gradient regularization (IGR, from earlier work, Smith et al. 2021).** On average, SGD follows the gradient flow of a modified loss = loss + (learning rate / 4) * mean squared norm of the per-example loss gradient with respect to the parameters
2. **Step for common losses.** For least squares the per-example gradient is `residual * grad_theta f`. Thus the implicit penalty is on `||grad_theta f(x)||^2`, weighted by the squared residual. The penalty is stronger when residuals are large, for example early in training
3. **Theorem 1 (parameters -> inputs).** For a layered network, `||grad_x f(x)||^2` multiplied by a positive factor is at most `||grad_theta f(x)||^2`. Thus a penalty on the parameter gradient is also a pressure on the input gradient
4. **Conclusion.** SGD puts an implicit pressure on the discrete Dirichlet energy, which is the proxy for the geometric complexity. The paper names this the Geometric Occam's Razor
5. **When is the pressure strong?** When the quantities `A_i = ||w_i|| * ||h_i'(x)||` stay small. The paper expects this early in training, when the spectral norms are near zero. Without spectral regularization the norms usually grow during training
6. **Remark 2.** For one linear layer, the Dirichlet energy is equal to the squared L2 norm of the parameters. This recovers the known bias of SGD to low-L2-norm solutions for linear models

### Equations

#### Geometric complexity (volume of the graph)

`Omega_D(f) = integral over X_D of sqrt(1 + ||grad_x f||^2) dx`

- In 1-D this is the arc length of the graph

#### First-order approximation, `sqrt(1 + z) ~ 1 + z / 2`

`Omega_D(f) ~ Vol(X_D) + DE(f)`

`DE(f) = (1/2) * integral over X_D of ||grad_x f||^2 dx`

#### Discrete Dirichlet energy

```python
DE_hat = (1 / (2 * len(D))) * sum(norm(grad(f(x), x)) ** 2 for x in D)
```

- `1 / len(D)` takes the place of the volume element

#### Modified loss of SGD (IGR)

```python
L_tilde = L + (h / (4 * len(D))) * sum(norm(grad(L(x, theta), theta)) ** 2 for (x, y) in D)
```

- `h` - learning rate. It is the implicit regularization rate
- The formula in the paper has no explicit batch-size term

#### Least squares case

```python
eps = f(x, theta) - y                      # signed residual
grad(L(x, theta), theta) == eps * grad(f(x, theta), theta)
L_tilde = L + (h / (4 * len(D))) * sum(eps ** 2 * norm(grad(f(x, theta), theta)) ** 2 for (x, y) in D)
```

#### Theorem 1

Network: `f_theta(x) = f_l(... f_1(x))`, `f_i(z) = a_i(w_i z + b_i)`. `h_i(x)` is the sub-network from the input `x` to layer `i`.

`||grad_x f(x)||^2 * sum_{i=1..l} (1 + ||h_i(x)||^2) / (||w_i||^2 * ||h_i'(x)||^2) <= ||grad_theta f(x)||^2`

- `||w_i||` - spectral norm of the weight matrix of layer `i`
- Lemma 3 (weights): `||grad_x f||^2 * (||h_i(x)|| / (||w_i|| * ||h_i'(x)||))^2 <= ||grad_{w_i} f||^2`
- Lemma 4 (biases): `||grad_x f||^2 * (1 / (||w_i|| * ||h_i'(x)||))^2 <= ||grad_{b_i} f||^2`
- Proof idea: a small change of the input has the same effect as some small change of the weights or biases of a layer. Then use the chain rule, and add the terms for all layers

### Model

- 1-D regression: ReLU network, 3 layers, 300 units for each layer, SGD, no explicit regularization, 10 data points
- CIFAR-10: ResNet-18
- Learning rate values, batch size, number of seeds, optimizer settings: not mentioned

## How is this paper novel?

- Defines the complexity of a model through the geometry of its graph (arc length / volume), not through the parameter count
- Shows that the Dirichlet energy is a first-order proxy for this complexity and that it is easy to estimate on the data
- Theorem 1 connects two worlds: regularization in parameter space (gradient norm with respect to weights) and smoothness in input space (gradient norm with respect to inputs)
- Explains with one mechanism why SGD with a larger learning rate finds smoother functions
- Connects to known ideas: harmonic functions and Dirichlet's principle, splines, minimum description length, gradient penalties, spectral normalization (discussion only, no experiments)

## List of experiments

- **1-D regression (qualitative, no numbers):** the learned function is shown at steps 1, 20, 40, 50, 5000, 10000, 20000, 30000. At the end of training, the arc length of the learned function is near the length of the shortest path that interpolates the data points
- **CIFAR-10, ResNet-18:** the discrete Dirichlet energy and the IGR term `||grad_theta L||^2` are measured at the time of maximum test accuracy, for a range of learning rates. Each point is one fully trained model. The two quantities decrease when the learning rate increases

### Ablation Studies

- Learning rate sweep only
- No experiment with an explicit regularizer (Jacobian penalty, spectral norm, flatness). These are in the related-work discussion only

### Efficiency analysis

Not mentioned

## Preliminaries

### Dirichlet energy

A measure of how much a function changes on a domain.

`DE(f) = (1/2) * integral ||grad f||^2 dx`

- Constant function: energy 0
- A function with the smallest Dirichlet energy for given boundary values is a harmonic function (Dirichlet's principle)

### Arc length and volume of a graph

- Graph of `f: R -> R` between `a` and `b`: `length = integral_a^b sqrt(1 + f'(x)^2) dx`
- Graph of `f: R^d -> R`: `volume = integral sqrt(1 + ||grad f||^2) dx`
- A flat function gives the smallest value (the volume of the domain). Each slope adds to it

### Implicit gradient regularization

- Gradient descent uses steps of finite size `h`, so it does not follow the exact gradient flow of the loss
- It follows the flow of a slightly different loss: `L + (h / 4) * ||grad L||^2` (for SGD, the per-example version shown above)
- Thus a larger learning rate pushes the solution to flatter regions of the loss, with no explicit regularizer

## GPU hours

Not mentioned

## Key takeaways

- Complexity of a network = how much its graph bends, not how many parameters it has
- SGD has a built-in pressure to keep the graph short / flat. The pressure scales with the learning rate
- In 1-D, the trained ReLU network is near the shortest curve through the data points
- A penalty on gradients with respect to parameters also bounds gradients with respect to inputs (Theorem 1)
- The pressure is strongest early in training, when residuals are large and spectral norms are small
- Evidence is small: one toy figure and one learning-rate sweep on CIFAR-10. It is a workshop paper

## What I still do not understand?

## Ideas to pursue

- **In common with the Squiggly Line Conjecture:** the same object and the same formula. The paper also sees the network as its graph in the space where the data points are, and its complexity measure is the conjecture's arc length `l(f) = integral sqrt(1 + |grad f|^2)`. The 1-D experiment uses a ReLU network, so the curve is piecewise linear, as in the deck
- **Where it contradicts and where it supports:**
  - Contradicts "more length is better": SGD pushes the length down, not up. The trained function is near the shortest curve that goes through the data points. A good solution is the least squiggly curve that still touches all points
  - Contradicts the coverage argument: to get `L(f, D) = 0` the curve does not have to fill the space with an epsilon-tube. It must only go through the `|D|` data points. The necessary length is the length of the shortest interpolating path, which is finite and depends on the data, not on `eps -> 0`
  - The paper states that the Occam's razor is not a parameter count. It gives no result that a larger network has a longer or shorter graph
  - Partial support: the "wasted length" part of the conjecture (loops that overlap give no coverage) agrees with the paper. Extra length that does not help to reach a data point is removed by SGD
  - The paper is about trained networks. It says nothing about the expected loss of randomly initialised functions, which is the literal claim of the conjecture
- **What the idea can borrow:**
  - The discrete Dirichlet energy as a cheap, differentiable proxy for `l(f)` in high dimension, and the feature polytope `X_D` as the domain (not the full cube `[-1, 1]^d`)
  - A two-phase restatement to test: large random networks start with enough "reach" to come near all data points, then SGD shortens the curve to the shortest interpolating path. Measure `DE_hat` and `L(f, D)` from initialization to the end of training for different sizes
  - The learning rate as a control knob for the length of the trained curve (larger learning rate -> lower Dirichlet energy)

## Similar papers

- [On the Expressive Power of Deep Neural Networks](https://arxiv.org/abs/1606.05336)
- [Deep ReLU Networks Preserve Expected Length](https://arxiv.org/abs/2102.10492)
- [A Universal Law of Robustness via Isoperimetry](https://arxiv.org/abs/2105.12806)
- [How many degrees of freedom do we need to train deep networks: a loss landscape perspective](https://arxiv.org/abs/2107.05802)
- [Loss Landscapes are All You Need (ICLR 2023)](https://openreview.net/forum?id=QC10RmRbZy9)
- [What's Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/abs/1911.13299)
- [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555)
- [Large language models implicitly learn to straighten neural sentence trajectories](https://arxiv.org/abs/2311.04930)
- [Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
