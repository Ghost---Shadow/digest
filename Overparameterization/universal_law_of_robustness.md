# [A Universal Law of Robustness via Isoperimetry](https://arxiv.org/abs/2105.12806)

## Meta

* Journal   - NeurIPS
* Year      - 2021
* Author    - Microsoft Research, Stanford University (Sébastien Bubeck, Mark Sellke)
* Code      - Not mentioned (theory paper, no code)
* Slides    - 
* One liner - To fit n noisy points in d dimensions smoothly, a model needs about n*d parameters: Lip(f) >= ~sqrt(n*d/p). Overparameterization is necessary for robustness.
* Model     - None (theory for Lipschitz-parametrized function classes)
* Datasets  - None used. The paper only discusses published numbers for MNIST and ImageNet
* Baselines - None (theory paper)

## What problem does it solve?

- Classical rule: to fit `n` equations you need about `n` unknowns. So `p ≈ n` parameters are enough to interpolate `n` data points
- In practice, networks have many more parameters than `n`. Why is this necessary?
- The paper gives one answer: to fit the data *smoothly* (small Lipschitz constant), you need about `n * d` parameters, where `d` is the data dimension
- A smooth fit is a robust fit. A small change of the input cannot change the output much. So the paper is a size-versus-robustness tradeoff
- It gives a proof (for bounded weights) of a tradeoff that Bubeck, Li and Nagaraj conjectured for two-layer networks

## How does it solve it?

### Setup

1. Data: `n` i.i.d. pairs `(x_i, y_i)`, with `x_i` in `R^d` and `y_i` in `[-1, 1]`
2. Function class `F = {f_w}` with `p` real parameters. The weights are in a set of diameter `W`. `W` is at most polynomial in `n, d`
3. The parametrization is Lipschitz: `||f_w1 - f_w2||_inf <= J * ||w1 - w2||`
4. The distribution of `x` satisfies isoperimetry (or is a mixture of `k` such distributions)
5. The labels have noise: `sigma^2 = E[Var(y | x)] > 0`
6. "Fit below the noise level" means: training error `<= sigma^2 - eps`. The function then memorizes some of the label noise

### Main results

- **Theorem 1 (informal):** with high probability, for all `f` in `F` at the same time:

  `(1/n) * sum((f(x_i) - y_i)^2) <= sigma^2 - eps   =>   Lip(f) >= ~ (eps / sigma) * sqrt(n * d / p)`

- **Formal version:** the bound is

  `Lip(f) >= (eps / (sigma * sqrt(C * c))) * sqrt(n * d / (p * log(1 + 60 * W * J / eps) + log(4 / delta)))`

  - `c` - isoperimetry constant
  - `C` - an absolute constant
  - `delta` - failure probability
  - Needs `d` larger than a constant times `c * sigma^2 / eps^2`, and `k` (number of mixture components) small compared to `n * eps^2`
- **Sparse version:** if each `w` has at most `s` non-zero entries, `p * log(...)` becomes `s * log(p * ...)`
- **Reading of the law:**
  - `p ≈ n` -> `Lip(f) >= sqrt(d)`. The fit is possible but not smooth
  - `p ≈ n * d` -> `Lip(f) = O(1)` becomes possible. Smooth fit needs `d` times more parameters than plain fit
- **Deep networks (Lemma 3.1):** for a network of depth `D` with weights in `[-W, W]`, `J` can be `R * (W * Q * p)^D`. `Q` is the maximum number of matrix entries tied to one parameter. Only `log J` is in the bound, so for constant depth the law is `sqrt(n * d / (D * p))` up to log factors
- **Generalization view:** the same proof gives a bound on Rademacher complexity for a finite class of `L`-Lipschitz functions: `Rad <= O(max(sqrt(k / n), L * sqrt(c * log|F| / (n * d))))`
- **Tightness:**
  - `f(x) = sum(g(||x - x_i||) * y_i)` (one bump for each point) is `O(1)`-Lipschitz with `p = n * (d + 1)` parameters
  - For `p` between `n` and `n * d`: project the data to dimension `p / n` first. This gives `Lip = O(sqrt(n * d / p))`
- **Each assumption is necessary:**
  - Bounded weights: without it, one parameter can encode all functions. Appendix A gives a three-layer network with one parameter that memorizes random labels and stays Lipschitz
  - Noise `sigma^2 > 0`: without noise, a smooth perfect function could exist
  - Isoperimetry: a weaker property (Poincaré inequality) gives only a weaker bound

### Equations

#### Isoperimetry (Definition 1.1)

`P[|f(x) - E[f]| >= t] <= 2 * exp(-d * t^2 / (2 * c * L^2))` for each bounded `L`-Lipschitz `f`

- An `L`-Lipschitz function of `x` is almost a constant. Its fluctuation is about `L / sqrt(d)`
- Examples: Gaussian `N(0, I_d / d)`, uniform on the sphere, uniform on the hypercube, strongly log-concave measures, manifolds with positive Ricci curvature

#### Proof sketch

```python
# 1. One fixed L-Lipschitz function f. Labels have noise z_i.
#    To fit below the noise level, f must correlate with the noise:
#    (1/n) * sum(f(x_i) * z_i) >= eps / 4
# 2. By isoperimetry, f(x_i) is concentrated near its mean (width L / sqrt(d)),
#    so it cannot follow independent noise
P_one_f_fits = exp(-n * d * eps**2 / L**2)        # up to constants

# 3. Union bound on a finite class of N functions
P_some_f_fits = N * P_one_f_fits

# 4. A class with p bounded parameters has an eps-net with
log_N = p * log(W * J / eps)

# 5. The union bound fails to exclude a fit only if
#    log_N >= n * d * eps**2 / L**2
#    =>  L >= eps * sqrt(n * d / (p * log(W * J / eps)))
```

- The trade is between the number of functions the class can express (`exp(p)`) and the probability that one smooth function fits (`exp(-n * d / L^2)`)

### Model

- No trained model. The result is for all classes of functions with a Lipschitz parametrization and bounded weights
- Feed-forward networks (with skip connections and weight sharing) with 1-Lipschitz non-linearities are an instance

## How is this paper novel?

- The first proof of the size-versus-smoothness tradeoff for a general class of models. Earlier work had it only as a conjecture for two-layer networks with Gaussian data (Bubeck, Li, Nagaraj)
- The proof does not use the structure of the network. It uses only the number of parameters, the size of the weights, and the concentration of the data
- It includes mixtures of isoperimetric distributions, which is nearer to real data
- It gives a new reason for overparametrization: the reason is not optimization. Smooth interpolation needs `n * d` parameters

## List of experiments

- No experiments of the authors. The paper compares the law with published results
- **MNIST (numbers from Madry et al.):** `n = 6 * 10^4`, `d = 784`
  - Robust accuracy goes from about 10% to about 90% near `2 * 10^5` parameters, and to about 95% near `3 * 10^6` parameters
  - With an effective dimension between 15 and 150, `n * d` is between `10^6` and `10^7`. This agrees with the transition
  - The rule `p ≈ n` gives only about `10^4` to `10^5`
- **ImageNet:** about `1.4 * 10^7` images
  - With an effective dimension much smaller than the pixel count, the law gives `10^10` to `10^11` parameters for a robust model
  - Current models are about `10^8` to `10^9` parameters. The authors say that robust ImageNet models are possibly some orders of magnitude too small
- The authors say these estimates are speculative

### Ablation Studies

- Not applicable. In place of ablations, the paper shows that each assumption is necessary (see "Main results")

### Efficiency analysis

Not mentioned

## Preliminaries

### Lipschitz constant

`|f(x) - f(y)| <= L * ||x - y||` for all `x, y`. The smallest such `L` is `Lip(f)`.

- For a differentiable `f`, `Lip(f) = max ||grad f||`
- Small `Lip(f)` = smooth = robust to small input changes
- It is a worst-case (maximum) quantity, not an average

### Isoperimetry / concentration of measure

In high dimension, a Lipschitz function of a random point is almost constant.

- Example: `x` uniform on the unit sphere in `R^d`. Each `L`-Lipschitz `f(x)` is in a window of width about `L / sqrt(d)` around its mean
- Consequence: a smooth function cannot give very different outputs to random points. To follow random labels, the function must have a large `L`

### Epsilon-net and union bound

- An eps-net is a finite list of functions such that each function of the class is within `eps` of one in the list
- A class with `p` bounded parameters has a net of size about `(W * J / eps)^p`
- Union bound: `P(some f in the net fits) <= size_of_net * P(one f fits)`

## GPU hours

Not mentioned (no experiments)

## Key takeaways

- Fit of `n` noisy points with `p` parameters forces `Lip(f) >= sqrt(n * d / p)`
- A smooth fit needs `p ≈ n * d`. Overparametrization by a factor `d` is the price of robustness
- With few parameters, the fit is still possible, but the function must be steep somewhere
- The law is about what the class can express. It says nothing about the optimizer or about robust generalization
- The law is for the `l2` norm. The `l_inf` version is only a belief of the authors
- The data dimension in the law is an effective dimension. For real data it is much smaller than the pixel count and it is hard to estimate

## What I still do not understand?

## Ideas to pursue

- **In common with the Squiggly Line Conjecture:** both connect the parameter count `p` to a geometric property of the graph of `f`. The law uses `Lip(f) = max |grad f|`. The conjecture uses the arc length `l(f) = integral of sqrt(1 + |grad f|^2)`, which is an average of the same gradient. Both use a counting argument: how many data points can a function family "reach"
- **Where it contradicts:** the direction is opposite for fitted functions. The law says that a *small* network must be steep (very squiggly) to fit the data, and a *large* network can fit the same data with a *smoother* function. So "more parameters -> more squiggly -> lower loss" is not what the law gives. In the proof, parameters and steepness are substitutes: the fit needs `p * Lip^2 >= n * d` (up to logs). More parameters decrease the necessary squiggliness. The conjecture is about random initialization and a distance-to-graph loss, so the law does not directly disprove it, but it shows that after training the larger network is not the longer curve
- **What the idea can borrow:**
  - The product `p * Lip^2 >= n * d` as a model of "coverage budget". Replace the claim "more parameters give more length" with "coverage needs parameters or length, and one can pay for the other"
  - Isoperimetry as the tool to compute the distance of a random point to the graph. In high `d` a Lipschitz graph is almost flat for random `x`, so the tube volume formula `l(f) * k * eps^(d-1)` must include the concentration factor `exp(-d * t^2 / L^2)`
  - The `n * d` scaling as a test: measure `l(f)` and `Lip(f)` of trained networks at widths below and above `p = n * d` and check which one changes
  - The effective dimension. The exponent `1/d` in the conjecture must use the effective dimension of the data, not the pixel count

## Similar papers

- [On the Expressive Power of Deep Neural Networks](https://arxiv.org/abs/1606.05336)
- [Deep ReLU Networks Preserve Expected Length](https://arxiv.org/abs/2102.10492)
- [The Geometric Occam's Razor Implicit in Deep Learning](https://arxiv.org/abs/2111.15090)
- [How many degrees of freedom do we need to train deep networks: a loss landscape perspective](https://arxiv.org/abs/2107.05802)
- [Loss Landscapes are All You Need](https://openreview.net/forum?id=QC10RmRbZy9)
- [What's Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/abs/1911.13299)
- [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555)
- [Large language models implicitly learn to straighten neural sentence trajectories](https://arxiv.org/abs/2311.04930)
- [Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
