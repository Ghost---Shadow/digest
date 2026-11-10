# [How many degrees of freedom do we need to train deep networks: a loss landscape perspective](https://arxiv.org/abs/2107.05802)

## Meta

* Journal   - ICLR 2022
* Year      - 2021
* Author    - Stanford University, Meta AI (Brett W. Larsen, Stanislav Fort, Nic Becker, Surya Ganguli)
* Code      - [ganguli-lab/degrees-of-freedom](https://github.com/ganguli-lab/degrees-of-freedom)
* Slides    - 
* One liner - Training in a random affine subspace succeeds when its dimension is above a threshold. Gordon's escape theorem predicts the threshold from the size of the low-loss sublevel set.
* Model     - Conv-2, Conv-3, ResNet-20
* Datasets  - MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, SVHN
* Baselines - Random affine subspaces, burn-in subspaces, lottery subspaces, lottery tickets

## What problem does it solve?

- A network with `D` parameters can be trained with many fewer degrees of freedom. Examples: lottery tickets (pruning), and training in a random low-dimensional subspace of the weights
- Earlier work saw a threshold: training in a random subspace of dimension `d` fails for small `d` and succeeds above some `d`. There was no explanation for the threshold
- Questions of the paper:
  - What sets the threshold dimension?
  - Why does it depend on the initialization and on the target loss?
  - How do random subspaces, burn-in subspaces and lottery tickets compare?

## How does it solve it?

### Training flow

1. Select a start point `w_t` in `R^D`
   - Random affine subspace: `w_0` = the random initialization
   - Burn-in affine subspace: `w_t` = the weights after `t` steps of full training
2. Sample a matrix `A` of size `D x d`. The columns are Gaussian and normalized to unit length
3. Set `w(theta) = A @ theta + w_t`, with `theta` in `R^d` and `theta = 0` at the start
4. Train only `theta` with Adam. `A` and `w_t` are fixed
5. The run is a success if the loss gets to the target sublevel set `S(eps)` (or the target accuracy)
6. Repeat for many `d` and many samples of `A`. Record the success probability
7. The threshold training dimension is the smallest `d` with success probability `>= 1 - delta`

Lottery subspace (a third method)

1. Train the full network. Save the full trajectory `w_0 ... w_T`
2. Compute the top `d` principal components `U_d` of the trajectory
3. Train `w(theta) = U_d @ theta + w_t` from a rewound point `w_t`

### Inference

1. No change. After training, `w = A @ theta + w_t` is a normal weight vector of the full network

### Equations

#### Definitions

```python
S(eps) = {w : L(w) <= eps}                        # loss sublevel set
P_s(d, eps, t) = P[S(eps) intersects (w_t + span(A))]   # success probability
d_star(eps, t, delta) = min d such that P_s(d, eps, t) >= 1 - delta
```

#### Gaussian width

`w(S) = 0.5 * E[sup over x, y in S of <g, x - y>]`, with `g ~ N(0, I_D)`

- It measures the size of a set as seen by a random direction
- For a ball of radius `r` in a `d`-dimensional subspace: `w(S) ≈ r * sqrt(d)`

#### Local angular dimension

```python
proj(S, w_t) = {(x - w_t) / norm(x - w_t) for x in S}   # project S on the unit sphere around w_t
d_local(S, w_t) = gaussian_width(proj(S, w_t)) ** 2
```

- It is the size of the "shadow" of `S` as seen from the start point `w_t`
- A large and near set has a large shadow. A small or distant set has a small shadow

#### Gordon's escape theorem (Theorem 3.1) and main theorem (Theorem 3.2)

- `S` = closed subset of the unit sphere in `R^D`. `Y` = random subspace of dimension `d = D - k`
- If `k > w(S)^2`, then `P(Y does not intersect S) >= 1 - 3.5 * exp(-(k / sqrt(k + 1) - w(S))^2 / 18)`
- Theorem 3.2 is the same statement for an affine subspace through `w_t`, with `w(proj(S, w_t))` in place of `w(S)`

#### Threshold training dimension

`d_star(S(eps), w_t) = D - d_local(S(eps), w_t)`

- `d + d_local > D` -> the random subspace hits the sublevel set with high probability -> success
- `d + d_local < D` -> the random subspace misses the set -> failure
- This is a phase transition, not a slow change

#### Quadratic well

```python
L(w) = 0.5 * w.T @ H @ w                 # H has eigenvalues lambda_i
r_i = sqrt(2 * eps / lambda_i)           # radii of the ellipsoid S(eps)
R = norm(w_0)                            # distance from the start point to the minimum

d_local >~ sum(r_i**2 / (R**2 + r_i**2))
d_star  <~ D - sum(r_i**2 / (R**2 + r_i**2))
```

- Flat directions (small `lambda_i`, large `r_i`) add almost 1 to `d_local`. Sharp directions add almost 0
- Larger `eps` -> larger `r_i` -> smaller threshold
- Smaller `R` (start nearer to the minimum) -> smaller threshold

### Model

- Conv-2: 2 convolution layers (16, 32 channels), ReLU, max-pooling, one fully connected layer
- Conv-3: 3 convolution layers (32, 64, 64 channels)
- ResNet20v1 with batch norm that uses batch statistics
- Parameter counts on CIFAR-10: 25.6k (Conv-2), 66.5k (Conv-3), 272.5k (ResNet20). On MNIST / Fashion-MNIST: 20.5k, 61.5k, 272.2k
- Adam, learning rate 5e-2 for subspace training, batch size 128, 3 to 5 epochs
- Runs for each dimension: 10 (Conv-2), 5 (Conv-3), 3 (ResNet20). `delta` = 0.1, 0.2, 0.33

## How is this paper novel?

- It gives a theory of the threshold training dimension with a high-dimensional geometry result (Gordon's escape theorem)
- It introduces the local angular dimension. This one quantity explains the effect of the target loss and of the start point
- It shows that the threshold is a phase transition in the success probability
- It introduces lottery subspaces (principal components of the training trajectory)
- It puts random subspaces, burn-in subspaces, lottery subspaces and lottery tickets in one comparison

## List of experiments

- **Phase transition:** for each model and dataset, the success probability goes sharply from 0 to 1 when `d` increases. The exact threshold numbers are in the figures, not in the text
- **Target accuracy:** a higher target accuracy (lower loss) needs a larger threshold dimension
- **Burn-in:** at high target accuracy the order of the thresholds is `t = 16` < `t = 8` < `t = 4` < random affine (`t = 0`). More burn-in steps -> fewer dimensions
- **Train vs. test:** the trends are the same for train accuracy and test accuracy
- **Lottery subspaces:** get to the accuracy of the full model at about 20 to 60 dimensions. They are better than random subspaces and lottery tickets at low dimension
- **Lottery tickets:** 2 epochs of training, magnitude pruning, rewind to the initialization. Tested to a maximum of 4096 parameters. In the range where both exist, tickets are not better than random affine subspaces, but Conv-2 on CIFAR-10 is an exception. Tickets fail at high compression because of layer collapse

### Ablation Studies

- **Number of burn-in steps:** `t = 4, 8, 16` for Conv-2 and `t = 8, 16` for ResNet20
- **Linearized model (NTK):** for Conv-2 on MNIST and Fashion-MNIST, the model linearized around an optimum gives a threshold near that of the full model
- **Lottery subspace spectrum:** the singular values of the trajectory decrease quickly. CIFAR-10 shows a sharp accuracy change near `d = 10`. The authors connect this to the class structure of the Hessian
- **Accuracy at the start point:** the gain of burn-in increases with the dimension. Conv-3 on MNIST shows no gain from burn-in

### Efficiency analysis

- Not given as a table. The authors say that the lottery ticket runs were limited to 4096 parameters because of the compute cost

## Preliminaries

### Training in a random subspace (intrinsic dimension)

From Li et al. 2018. Do not train all `D` weights. Train only `d` numbers `theta`, and set `w = A @ theta + w_0` with a fixed random `A`.

- The smallest `d` that gives about 90% of the full accuracy was named the intrinsic dimension of the task
- This paper explains where that number comes from

### Gaussian width

`w(S) = 0.5 * E[max over x, y in S of <g, x - y>]`

- Take a random direction `g`. Measure the width of `S` along `g`. Take the mean
- `w(S)^2` behaves like a dimension. For a `k`-dimensional subspace cut by the unit sphere, `w^2 ≈ k`

### Intersection of subspaces

- Two generic subspaces of dimension `a` and `b` in `R^D` intersect (outside the origin) only if `a + b > D`
- Example: two lines in 3D do not meet (`1 + 1 < 3`). A line and a plane meet (`1 + 2 >= 3`)
- Gordon's theorem extends this rule to a set that is not a subspace. `w(S)^2` takes the place of the dimension of the set

## GPU hours

- Not mentioned. The paper only says that the runs used NVIDIA TITAN Xp GPUs on an internal cluster, with JAX and Flax

## Key takeaways

- Success of training in `d` random dimensions is a hit-or-miss problem: does a random `d`-dimensional slice through the start point hit the low-loss set?
- `d_star = D - d_local`. The necessary number of degrees of freedom is the codimension of the low-loss set as seen from the start point
- The low-loss set of an overparametrized network is very large (high local angular dimension). This is why a small `d` is sufficient
- A better start point (burn-in) and a less strict target loss decrease the threshold
- A data-dependent subspace (lottery subspace) needs only tens of dimensions
- Limits: the theory cannot compute `d_local` for a real network. The quadratic bound is crude. The models and datasets are small. The theory is about training loss, not about generalization

## What I still do not understand?

## Ideas to pursue

- **In common with the Squiggly Line Conjecture:** both are coverage arguments about a random object. Here: a random `d`-dimensional slice hits the low-loss set if `d + d_local > D`. In the conjecture: a random curve comes near the data points if its tube volume is large. Both say that more degrees of freedom increase the probability of a hit, and both use the size of a set as seen from the start point
- **Where it supports and where it contradicts:**
  - Support: more trainable dimensions give a higher probability of low loss, and the effect has a limit (above the threshold, more dimensions give nothing)
  - Contradiction: the size that is important is the size of the low-loss set in *weight space*, not the arc length of the function in *data space*. The paper never needs a more squiggly function. Also the transition is sharp (a phase transition at `d_star`). The conjecture gives a smooth law `(uncovered volume)^(1/d)` with slow diminishing returns. The measured curves in this paper are step-shaped
  - The paper is about training in a subspace, not about the loss of a random function at initialization
- **What the idea can borrow:**
  - Gordon's escape theorem as the rigorous tool for "a random object of dimension `k` hits a set". It can replace the informal tube-volume estimate
  - Gaussian width in place of volume. In high dimension, volume is a bad measure of size (the `eps^(d-1)` factor goes to zero). Width squared behaves like a dimension and gives clean thresholds
  - The dependence on the start point. Define the "local angular dimension of the dataset as seen from the graph of `f`" and check if it predicts the loss
  - The measurement protocol: success probability as a function of dimension, with a fixed `delta`. Use the same protocol to test the conjecture on toy problems

## Similar papers

- [On the Expressive Power of Deep Neural Networks](https://arxiv.org/abs/1606.05336)
- [Deep ReLU Networks Preserve Expected Length](https://arxiv.org/abs/2102.10492)
- [The Geometric Occam's Razor Implicit in Deep Learning](https://arxiv.org/abs/2111.15090)
- [A Universal Law of Robustness via Isoperimetry](https://arxiv.org/abs/2105.12806)
- [Loss Landscapes are All You Need](https://openreview.net/forum?id=QC10RmRbZy9)
- [What's Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/abs/1911.13299)
- [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555)
- [Large language models implicitly learn to straighten neural sentence trajectories](https://arxiv.org/abs/2311.04930)
- [Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
