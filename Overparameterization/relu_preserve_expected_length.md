# [Deep ReLU Networks Preserve Expected Length](https://arxiv.org/abs/2102.10492)

## Meta

* Journal   - ICLR 2022 (the arXiv page does not name a venue)
* Year      - 2021
* Author    - Princeton University, University of Pennsylvania, McGill University & Mila (Hanin, Jeong, Rolnick)
* Code      - Not mentioned
* Slides    - 
* One liner - At standard (He) initialisation the expected length distortion of a ReLU network does not grow with depth. It shrinks slightly.
* Model     - Fully connected ReLU networks at initialisation
* Datasets  - None (random networks at initialization; the input is a fixed line segment)
* Baselines - Lower bounds of Raghu et al. (2017)

## What problem does it solve?

- Common belief: a random deep network makes a curve exponentially longer with depth (Raghu et al. 2017; Price and Tanner)
- The paper asks: take a curve `M` of length 1 in input space. What is the length of the output curve `N(M)` for a ReLU network with the standard initialization?
- Answer: the expected length does not grow with depth. It decreases a little
- The older results are lower bounds only, and their exponential growth comes from weights that are larger than the standard ones
- The paper also reports a typo in Theorem 3(a) of Raghu et al.: a factor 2 is missing. The correct lower bound is `Omega( (sigma_w * sqrt(width) / (2 * sqrt(width + 1)))^depth )`. Without the 2 the statement is false for He initialization (`sigma_w = sqrt(2)`). The typo was copied into Theorem 1 of Price and Tanner

## How does it solve it?

### Setup

1. Fully connected ReLU network `N` with input dimension `n_0`, hidden widths `n_1 ... n_(L-1)`, output dimension `n_L`, depth `L`
2. He normal initialization: weights `W_l[i][j] ~ N(0, 2 / n_(l-1))` (variance 2 / fan-in), biases `b_l[j] ~ N(0, C_b)` with any fixed `C_b > 0`
3. `M` is a smooth curve of length 1 in `R^(n_0)`, with unit-speed parameterization `gamma(t)`, `t in [0, 1]`
4. Output curve: `Gamma = N(gamma)`. `len(N(M)) = integral_0^1 || Gamma'(t) || dt`
5. Length distortion = `len(N(M)) / len(M)`
6. Scaling remark: multiply all weights and biases by `c` (variance `2 * c^2 / fan-in`) and the distortion is multiplied by `c^L`. Thus exponential growth or decay is expected for any scale that is not the He scale

### Main results

1. **Mean (Theorem 3.1 / 5.1):** the expected length distortion is at most `sqrt(n_L / n_0)`. It has a closed-form expression, which decreases with the sum of `1 / n_l` over the hidden layers
2. **Variance (Theorem 4.1 / 5.1):** `Var[len(N(M))] <= n_L / n_0`
3. **Higher moments (Theorem 4.1 / 5.1):** all moments have an upper bound with the same shape. The paper says this points to an upper bound of log-normal type
4. **Volumes (Theorem 5.2):** the same is true for a `d`-dimensional manifold `M` of volume 1. The mean and the variance of the `d`-volume of `N(M)` do not grow with depth
5. **Extension:** the results hold for any symmetric weight distribution with variance 2 / fan-in and finite higher moments (the constant 5/8 can change)

### Equations

#### Mean length (eq. 5.1)

```python
C = gamma_fn((n_L + 1) / 2) / (gamma_fn(n_L / 2) * sqrt(n_L / 2))     # close to 1, depends on n_L only
E_len = sqrt(n_L / n_0) * C * exp(-(5 / 8) * sum(1 / n[l] for l in range(1, L))
                                  + O(sum(1 / n[l] ** 2 for l in range(1, L))))
```

- `gamma_fn` - the Gamma function
- The sum is over the hidden layers `1 ... L-1`
- Infinite width: the exponent goes to 0
- Constant width `n`: the exponent is about `-5 * L / (8 * n)`. The mean decreases with depth, and decreases faster for narrow networks
- The factor `sqrt(n_L / n_0)` comes from `E ||N(x)||^2 ~ (n_L / n_0) * ||x||^2`
- Example in the paper: `n_0 = n_L = 5` gives the infinite-width value `Gamma(3) / (Gamma(2.5) * sqrt(2.5)) ~ 0.9515`

#### Variance (eq. 5.2)

`Var[len(N(M))] <= n_L / n_0`

#### Higher moments (eq. 5.3)

`E[len(N(M))^m] <= (n_L / n_0)^(m/2) * exp(c2 * m^2 * sum_{l=1..L} 1 / n_l)`   if `m < c1 * min(n_1 ... n_(L-1))`

- `c1`, `c2` - universal constants. Their values are not given

#### Volume of a d-dimensional manifold (Theorem 5.2)

`(E[vol_d(N(M))])^2` and `Var[vol_d(N(M))]` are both `<= (n_L / n_0)^d * exp(-(d * (d - 1) / 2) * sum_{l=1..L} 1 / n_l)`

- Condition: `d <= min(n_0 ... n_L)`
- Appendix: `E || J u_1 ^ ... ^ J u_d ||^2 = (n_L / n_0)^d * prod_{l=1..L} prod_{j=1..d} (1 - (j - 1) / n_l)`

#### Proof steps

1. **Lemma 6.1:** `E[len(N(M))^m] <= integral_0^1 E[ || J_gamma(t) gamma'(t) ||^m ] dt`. A statement about a full curve becomes a statement about the input-output Jacobian `J` at one point
2. **Proposition 6.2:** for a fixed input `x` and a unit vector `u`, `||J_x u||^2` has the same distribution as a product of independent terms, one for each layer:

```python
# n_on[l] ~ Binomial(n[l], 1/2): number of ReLU units that are "on" in layer l
J_u_sq = (n_L / n_0) * prod((2 / n[l]) * chi2(n_on[l]) for l in range(1, L)) * (1 / n_L) * chi2(n_L)
```

3. Each hidden layer gives a factor with mean `1 - (5/8) * (1 / n_l) + O(1 / n_l^2)` for `||J u||`. The product of the factors gives the exponential with 5/8

### Model

- Fully connected ReLU networks at initialization only. No training
- Experiments: `n_0 = n_L = 5`, hidden width constant or mixed (20; 30 and 10 alternating; 30 then 10; 10 then 30), several depths, bias variance 0.1
- Appendix (preliminary): conv network (input 16x16x3 = 768, output 5, 3x3 kernels, no pooling, one final fully connected layer) and residual network (blocks are two-layer ReLU networks with scale `eta_l = 1 / L`)

## How is this paper novel?

- First upper bounds on length distortion. Older work has lower bounds only
- Works at the initialization that people use in practice (He), not at large weight scales
- Gives a closed-form expression for the mean, plus bounds on the variance and on all higher moments
- Extends from curves to `d`-dimensional volumes
- Lemma 6.1 (curve moments from one-point Jacobian moments) is new, as the authors believe
- Finds and corrects a typo in the well-known lower bound of Raghu et al.

## List of experiments

- **Mean distortion vs. depth (Figure 1):** `n_0 = n_L = 5`, constant hidden width, fixed unit line segment, 500 initializations for each architecture. The empirical means agree with Theorem 5.1. The lower bounds of Raghu et al. are "quite loose". The distortion does not grow with depth, decreases a little, and decreases faster for narrow networks
- **Distortion vs. `n_L / n_0` (Figure 2):** 100 pairs `(n_0, n_L)`, maximum dimension 50, 200 initializations for each pair. The log of the mean follows the line `log y = 0.5 * log x`, thus distortion scales as `sqrt(n_L / n_0)`
- **Mean and standard deviation vs. `sum(1 / n_l)` (Figure 3):** four architecture types, `n_0 = n_L = 5`, 100 initializations for each network. The mean decreases with `sum(1 / n_l)`. The standard deviation does not usually increase with depth
- **Conv network (Appendix A):** mean distortion is approximately `sqrt(n_L / n_0) ~ 0.08` and decreases a little with depth
- **Residual network (Appendix A):** distortion decreases a little with depth, with a sharper decrease at small depth. 100 initializations

### Ablation Studies

- Width: 4 architecture types with the same or different `sum(1 / n_l)`
- Depth: several depths for each width
- Ratio of output dimension to input dimension
- Architecture family: fully connected, convolutional, residual (the last two are preliminary and without theory)

### Efficiency analysis

Not mentioned. The length is computed exactly: find the points where the line segment crosses a boundary between linear regions (binary search with 15 iterations, then solve linear equations), then add `||W_i w|| * (t_(i+1) - t_i)` over the linear pieces.

## Preliminaries

### He initialization

Weights have mean 0 and variance `2 / fan_in`.

```python
W = randn(fan_out, fan_in) * sqrt(2 / fan_in)
```

- ReLU sets about half of the units to 0. The factor 2 compensates for this
- Result: `E ||N(x)||^2` stays about the same from layer to layer. Activations and gradients do not explode or vanish
- In the notation of Raghu et al., He initialization is `sigma_w^2 = 2`

### Input-output Jacobian

`J_x` is the matrix of partial derivatives of the output with respect to the input at point `x`.

- For a ReLU network: `J_x = W_L * D_(L-1) * W_(L-1) * ... * D_1 * W_1`, where `D_l` is a diagonal 0/1 matrix (which units are on)
- A small step `u` in input space becomes the step `J_x u` in output space
- Thus the length of the output curve is the integral of `||J u||` along the input curve

### Chi-squared distribution

`chi2(k)` = sum of the squares of `k` independent standard normal numbers. Mean is `k`.

- The squared norm of a Gaussian vector with `k` entries is `chi2(k)`
- In the paper, each layer multiplies the squared length of a step by `(2 / n) * chi2(number of units that are on)`

## GPU hours

Not mentioned

## Key takeaways

- With He initialization, a random ReLU network does not make a curve longer. Expected distortion is about `sqrt(n_L / n_0)`, multiplied by a factor a little smaller than 1
- More depth makes the expected length a little shorter: factor `exp(-(5/8) * sum(1 / n_l))`
- More width cancels this decrease, but the length cannot go above the limit `sqrt(n_L / n_0)`
- Exponential growth in older papers is an effect of the weight scale: scale `c` gives the factor `c^L`
- The same holds for areas and volumes, not only for curves
- All results are for initialization. The paper says nothing about trained networks, and has no proof for conv / residual networks
- The authors think that curvature can behave in a similar way, but give no proof

## What I still do not understand?

## Ideas to pursue

- **In common with the Squiggly Line Conjecture:** the paper studies exactly the central quantity of the conjecture, the expected arc length of a curve made by a randomly initialised ReLU network, as a function of depth and width. Theorem 5.2 does the same for `d`-dimensional volumes, which is the correct object for the graph of `f: [-1, 1]^d -> R` (the graph is a `d`-dimensional surface, not a curve)
- **This paper contradicts the conjecture in its "random initialization" form:**
  - At standard (He) initialization, a network with more parameters is not more squiggly. The expected length has the upper limit `sqrt(n_L / n_0)` for all depths and widths
  - More depth makes the curve a little shorter, not longer
  - For a scalar output, the length of the graph of `f` along a unit segment is `integral sqrt(1 + |f'|^2) <= 1 + len(N(M))` (my step, not in the paper). Thus the expected graph length is at most `1 + sqrt(1 / n_0)`, independent of the parameter count. The epsilon-tube volume `l(f) * k * eps^(d-1)` then cannot grow with the size of the network
  - Small support: more width removes the factor `exp(-(5/8) * sum(1 / n_l))`, so a wider network is a little longer than a narrow one of the same depth. But this effect saturates and is far from "covers more of the space"
  - The conjecture can survive only if (a) the weight scale is larger than He (then the `c^L` factor gives growth with depth, as in Raghu et al.), or (b) it is stated for trained networks, about which this paper says nothing
- **What the idea can borrow:**
  - The closed-form mean and the moment bounds as the baseline `E[l(f)]` at initialization. Put this into the tube-volume formula to get a concrete, testable prediction of expected `L(f, D)` versus width and depth
  - The exact method to measure length for a piecewise linear network (breakpoints of the linear regions), and the experiment design (hundreds of initializations for each architecture)
  - Theorem 5.2 for `d > 1`, and the `c^L` scaling rule as a control knob: change the weight scale, measure length and `L(f, D)` together, and test if lower loss follows length or follows parameter count

## Similar papers

- [On the Expressive Power of Deep Neural Networks](https://arxiv.org/abs/1606.05336)
- [The Geometric Occam's Razor Implicit in Deep Learning](https://arxiv.org/abs/2111.15090)
- [A Universal Law of Robustness via Isoperimetry](https://arxiv.org/abs/2105.12806)
- [How many degrees of freedom do we need to train deep networks: a loss landscape perspective](https://arxiv.org/abs/2107.05802)
- [Loss Landscapes are All You Need (ICLR 2023)](https://openreview.net/forum?id=QC10RmRbZy9)
- [What's Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/abs/1911.13299)
- [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555)
- [Large language models implicitly learn to straighten neural sentence trajectories](https://arxiv.org/abs/2311.04930)
- [Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
