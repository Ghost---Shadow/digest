# [The Puzzling Success of Overparameterization: Lottery Tickets or Escape Dimensions?](https://infoscience.epfl.ch/entities/publication/9a49779b-f9f8-448d-b3d1-737c78455309)

## Meta

* Journal   - Preprint (EPFL Infoscience). Position paper, NeurIPS 2026 poster
* Year      - 2026
* Author    - EPFL (Flavio Martinelli, Johanni Brea, Wulfram Gerstner)
* Code      - Not mentioned. The simulations use the MLPGradientFlow package (Brea et al., 2023)
* Slides    - 
* One liner - Position: width helps because it adds dimensions through which optimisation escapes bad minima, not because a larger network has a better chance of a winning initialisation.
* Model     - Two-layer MLPs (3 to 7 hidden neurons), Conv-6
* Datasets  - Synthetic teacher-student regression (2D input), CIFAR-10, MNIST (teacher-student setup from Martinelli et al., 2024)
* Baselines - The lottery ticket explanation

Note: this digest is written from the full text of the EPFL preprint (dated May 12, 2026, [PDF on the page of the first author](https://flavio-martinelli.github.io/assets/pdf/EDT.pdf)). The NeurIPS 2026 poster has the title ["Position: Lottery Tickets Do Not Explain Overparameterization. How About Escape Dimensions?"](https://neurips.cc/virtual/2026/poster/140102). I did not compare the NeurIPS version with the preprint. No arXiv version was found.

## What problem does it solve?

- Common explanation in the field: "a larger network is easier to train because it contains more subnetworks, and each subnetwork is a lottery ticket. More tickets -> higher chance of a winning ticket"
- The paper says that this analogy is misleading. It does not attack the Lottery Ticket Hypothesis (LTH) itself, which is an empirical result about pruning
- It attacks the Lottery Ticket Conjecture (LTC): the untested claim of Frankle & Carbin that SGD finds and trains a well-initialized subnetwork, and that this is the reason why dense networks are easier to train
- The analogy implies three properties. The paper tests them
  - **Sufficiency:** one winning subnetwork at initialization is sufficient for success of the full network
  - **Scaling:** the probability of success increases combinatorially with the size of the network
  - **Independence:** the result of one subnetwork does not depend on the other weights
- If the properties were true, training a wide network would be the same as many independent optimizations in parallel (multi-start optimization)
- The paper proposes a different mental picture: **Escape Dimensions**

## How does it solve it?

### Setup

Experiment 1: is a winning ticket sufficient? (teacher-student)

1. Teacher: two-layer ReLU MLP with 4 hidden neurons, 2D input. Data: 30000 samples, uniform on `[-sqrt(3), sqrt(3)]^2`
2. Train 100 students with 4 hidden neurons (same architecture as the teacher). MSE loss, gradient flow with an ODE solver
3. Success = final loss below `10^-25`. 7 of 100 runs are a success. Their initializations are the "winning tickets" `I*`
4. Put each winning ticket in a larger network: the first 4 neurons are `I*`, and `r - 4` new neurons get a random Glorot normal initialization (30 seeds), `r` in `{5, 6, 7}`
5. Train the larger network with the same procedure. Count the successes

Experiment 2: adversarial overparameterization (advOP) on CIFAR-10

1. Find lottery tickets of Conv-6 with iterative magnitude pruning (21 rounds, prune rate 0.2, 30 epochs for each round, 5 seeds)
2. Regrow a random fraction of the pruned connections (growth ratio 0.5)
3. Initialize the regrown weights `theta_adv` such that the first gradient of the ticket is opposite to the gradient of the ticket in isolation (5000 optimization steps on the cosine similarity)
4. Train normally. Compare with the ticket alone (LTH), with random regrown weights (rndOP), and with random sparse networks (rnd)

Experiment 3: does the scaling law hold?

1. Same type of teacher-student task (teacher with 4 hidden neurons, 1600 samples)
2. Train 4000 students with random initialization for each width `m` in `{4, 5, 6, 7}`
3. Measure the probability of failure. Compare with the prediction of independent subnetworks

Experiment 4: show an escape dimension

1. Train a student with 3 hidden neurons (softplus) on a teacher with 4 hidden neurons. It converges to a local minimum with MSE about `10^-4`
2. Start again from the same initialization, but split neuron 3 into two copies (neurons 3 and 4) with output weights `gamma` and `1 - gamma` of the original, `gamma = 1/2 + eps`
3. The function of the network is the same at the start. Train and compare the loss curves

### Main results

- **Sufficiency fails.** With one added neuron (`m = 5`), a network that contains a winning ticket succeeds in only about 65% of the runs. A lottery would give 100%
- **Tickets depend on their context.** advOP decreases the CIFAR-10 test accuracy of the tickets at almost all sparsity levels. rndOP gives about the same accuracy as LTH. One counterexample (advOP) is sufficient to reject a property that must hold always
- **Scaling fails.** The measured probability of failure decreases much more slowly with the width than the combinatorial predictions `p0^N_D` and `p0^N_E`
- **Reason:** each neuron is in many subnetworks. The gradient of one subnetwork depends on the added neurons through the forward path and the backward path. So the subnetworks are not independent samples
- **Why the LTH is true but the LTC is not:** pruning after training gives two conditions. (i) The pruned weights go near zero during training. (ii) The remaining network can be trained alone. The LTC uses only (ii). The weights that are to be pruned must be initialized such that they converge to zero. If not, they prevent the convergence of the winning subnetwork
- **Escape Dimensions Theory:** adding parameters that look redundant increases the number of dimensions available to escape sub-optimal minima. This makes it easier for gradient descent to move through the loss landscape
- **Theorem 3.1 (Fukumizu & Amari 2000, informal):** in two-layer MLPs with scalar output, each critical point of a network of width `m` stays a critical point (a manifold of them) in each wider network `m+ > m`
- **Theorem 3.2 (informal):** each local minimum of the smaller network becomes a saddle in the larger network. (The paper writes saddle*: a manifold of non-strict saddles that can contain some plateau saddles)
- **Corollary 3.3 (Petzka & Sminchisescu 2021, informal):** the same holds if neurons are added to any layer of a deep feed-forward network
- **Hierarchy of critical points:** the landscape of a wide network contains all critical points of all smaller networks, and they collect escape dimensions. Each redundant neuron can add up to `d` new escape dimensions, where `d` is the number of parameters of one neuron
- **Global structure (Simsek et al. 2021):** the number of saddle subspaces and the number of global-minimum subspaces both increase combinatorially with the width. For width `m >~ 1.5 * r` (`r` = minimal width that solves the task) the global-minimum subspaces are in the majority
- **Picture of the landscape with increasing width:** (a) rugged, many bad local minima, few good ones -> (b) more minima at lower loss -> (c) linear subspaces of zero-loss solutions appear -> (d) in the interpolation regime, almost every initialization is near a zero-loss manifold
- **Where a winning ticket fits:** it is an initialization of a *narrow* network that is in one of the rare good basins of the rugged landscape. It is found by training a *wide* network (benign landscape) and then pruning the dimensions that are not necessary for convergence

### Equations

#### Adversarial initialization of the regrown weights

`theta_adv* = argmin over theta_adv of cos(grad_T L(theta_T, theta_adv), grad_T L(theta_T, 0))`

- `theta_T` - weights of the ticket
- `grad_T L(theta_T, 0)` - first gradient of the ticket in isolation
- `grad_T L(theta_T, theta_adv)` - first gradient of the ticket in the larger network
- The attack is weak: it changes only the first step. Then training is normal

#### Lottery scaling prediction

```python
# sufficiency: one subnetwork succeeds -> the full network succeeds
# so: full network fails -> all N subnetworks fail
# independence: the N results are independent
P_fail_full <= p0 ** N

N_D = comb(m, 4)            # count subnetworks by nodes (teacher has 4 neurons)
N_E = comb(m * d, 4 * d)    # count by edges, d = d_in + d_out
```

- `p0` - probability of failure at `m = 4`, measured with 4000 runs
- Without independence: `P_fail_full = product(P_fail_n) + Delta_N`. `Delta_N` contains all interaction terms. The experiment shows that `Delta_N` is the main term, not a small correction

#### Other "lottery" models (Appendix C), slowest decay to quickest decay

`p0^(m/r)  <  data  <  O(m^(r-1) * (1-s)^m)  <  p0^comb(m, r)  <  p0^comb(m*d, r*d)`

- `p0^(m/r)` - disjoint subnetworks
- `O(m^(r-1) * (1-s)^m)` - independent neurons, `s = (1 - p0)^(1/r)`
- The measured data is between the first two. Both decrease exponentially in `m`, not combinatorially

#### Neuron splitting

```python
# one neuron: out = a * act(w @ x)
# split into two copies with the same input weights w:
out = gamma * a * act(w @ x) + (1 - gamma) * a * act(w @ x)
# the function of the network does not change
# a critical point of the small network maps to a manifold of critical points of the large network
```

#### Escape dimension

`H @ e_i = lambda_i * e_i`, with `lambda_i < 0`

- `H` - Hessian of the loss at the critical point of the larger network
- `e_i` - a direction with negative curvature. The loss can decrease along it. This is an escape dimension

#### Ratio of saddles to global minima (Appendix D, formulas from Simsek et al.)

```python
G(r, m) = sum((-1) ** (r - i) * comb(r, i) * i ** m for i in range(1, r + 1))
R(r_star, m) = sum(G(r_star - k, m) for k in range(1, r_star)) / T(r_star, m)
```

- `G(r, m)` - number of ways to duplicate `r` neurons into `m` slots
- `T(r_star, m)` - number of global-minimum subspaces (duplication and zero-type neurons)
- `R` decreases quickly when `m / r_star` increases

### Model

- Two-layer MLPs (ReLU or softplus) with 3 to 7 hidden neurons, 2D input, one output
- Conv-6 (Frankle & Carbin): 6 convolution layers in three blocks (64, 128, 256 channels), two fully connected layers of 256 units, 10 classes
- Two-layer MLP teacher (30 hidden neurons) trained on MNIST, with two-layer MLP students (Appendix D)
- Training of the small networks: gradient flow with the ODE solver KenCarp58, then a second-order optimizer, to make sure that the run converged

## How is this paper novel?

- It separates the LTH (empirical, valid) from the LTC (a conjecture about the cause, not tested) and says that the field mixes the two
- It makes the lottery analogy formal (sufficiency, scaling, independence) and tests each property
- It gives the adversarial overparameterization experiment: a winning ticket fails if the other weights are selected against it
- It collects known theory results (Fukumizu & Amari, Simsek et al., Petzka & Sminchisescu, Belkin, Mei et al.) under one name and one intuitive picture: escape dimensions
- It documents how common the lottery explanation is, with a list of 40 quotes from papers and online sources

## List of experiments

- **Embedded winning ticket (Fig. 3a):** at `m = 5` the success rate is about 65%. The prediction of the lottery analogy is 100%. A fully random initialization of the same size has a lower success rate than the embedded ticket
- **advOP on Conv-6 / CIFAR-10 (Fig. 3b, Appendix B):** LTH: 5 runs. advOP and rndOP: 15 runs each. rnd: 8 seeds. The results are put in two bins of weights remaining: about 51%-13% and about 11%-3%
  - advOP is below LTH at almost all sparsity levels
  - At the highest sparsity levels advOP is better than LTH. The authors think that the added capacity compensates for the attack
  - rndOP is equal to LTH or a small amount better
  - rnd is the worst
- **Scaling (Fig. 3c):** the measured probability of failure for `m = 4 ... 7` stays many orders of magnitude above `p0^N_D` and `p0^N_E`
- **Escape through a split neuron (Fig. 6):** the small network (3 neurons) stops at MSE about `10^-4`. The large network (4 neurons, same function at the start) becomes slow at the same loss and then escapes to a much lower loss. The two copies of the split neuron follow the same path until the saddle and then separate
- **4-neuron student to 5-neuron student (Fig. E8):** the 5-neuron network escapes from the minimum of the 4-neuron network to a lower loss. It gets to a different local minimum, not to the global minimum
- **MNIST teacher-student (Fig. D6 right):** 50 students for each width. At small overparameterization there are two groups of solutions: few at low loss, many at higher loss. With more width, the low-loss group becomes more common
- **Literature search (Appendix A):** 3788 papers cite the LTH paper (13 Nov 2025), 2503 with an arXiv version. Keyword filter -> 3049 text blocks from 931 papers -> LLM filter -> 188 sentences -> manual selection. Final list: 29 quotes from papers and 11 from online sources

### Ablation Studies

- **rndOP vs. advOP:** random regrown weights do not decrease the accuracy of the ticket. Only adversarial regrown weights do. The authors say that rndOP does not prove the lottery mechanism: it shows only that these tickets are robust to random perturbations
- **Filtered vs. not filtered:** results are shown with and without the runs that have an early-stopping accuracy `<= 20%`
- **Other definitions of "ticket" (Appendix C):** sampling without replacement, disjoint subnetworks, independent neurons. None agrees with the data
- **Split neuron with corrected learning rate (Fig. E7):** the learning rate of the input weights of the split neurons is multiplied by `1 / gamma`. The paths of the small and large network are then identical until the saddle
- **`eps = 10^-15` vs. `eps = 0` in the split:** similar results. The authors think that machine precision is sufficient to separate the two copies

### Efficiency analysis

Not mentioned

## Preliminaries

### Lottery Ticket Hypothesis (LTH) vs. Lottery Ticket Conjecture (LTC)

- **LTH:** a dense network with random initialization contains a subnetwork that, trained alone from its original initialization, gets to the test accuracy of the full network in the same number of iterations or fewer. This is an empirical result. The subnetwork is found by pruning *after* training
- **LTC:** SGD finds and trains a well-initialized subnetwork. Dense networks are easier to train because they contain more possible subnetworks. Frankle & Carbin said that this is a conjecture and that they did not test it

### Critical points, local minima and saddles

- Critical point: the gradient of the loss is zero
- Local minimum: all eigenvalues of the Hessian are `>= 0`. Gradient descent stops here
- Saddle: one or more eigenvalues are negative. There is a direction along which the loss decreases, so gradient descent can go away (possibly slowly)

### Teacher-student setup

- A fixed "teacher" network makes the labels. A "student" network is trained to copy it
- If the student has the same width as the teacher or more, zero loss is possible. So "success" has a clear definition
- Overparameterization = width of the student / width of the teacher

## GPU hours

Not mentioned

## Key takeaways

- "A wide network wins because it has more lottery tickets" is not supported by the experiments of this paper. A winning ticket is not sufficient, and the gain from width is much slower than a combinatorial count
- Subnetworks are not independent. The other weights must also converge to a state in which they do not disturb the ticket
- The proposed picture: more width changes the *type* of the landscape. Bad minima of the narrow network become saddles in the wide network, and bad minima become rare compared to good minima
- The landscape of a wide network is not many copies of the landscape of a narrow network
- The paper is only about trainability. It does not address generalization, benign overfitting or implicit bias
- The authors do not claim that the lottery properties never hold. They say the analogy needs a rigorous new definition of "ticket"
- The theorems are informal statements of results from earlier papers. The new experiments are small (two-layer MLPs with 2D input, and Conv-6)

## What I still do not understand?

## Ideas to pursue

- **In common with the Squiggly Line Conjecture:** both want a geometric explanation of why more parameters help, in place of a count of subnetworks. Both say that more parameters give more "room". The diminishing returns in the conjecture agree in direction with the measured data here: the probability of failure decreases slowly with the width, much more slowly than a combinatorial law
- **Where it contradicts:**
  - The conjecture says that it connects to the lottery ticket hypothesis. This paper says that exactly this connection (a larger network has a better chance at initialization) is the wrong mental picture. The conjecture is a claim about the *function at initialization*. The paper shows that a property at initialization is not sufficient: a winning initialization fails in about 35% of the runs when one random neuron is added
  - Neuron splitting is a direct counterexample to arc length as the cause. The split network computes the *same function*. So the graph, the arc length `l(f)` and the coverage are identical. But the larger network escapes the minimum and the smaller one does not. The gain comes from new directions in *weight space*, not from a longer curve in *data space*
  - The paper puts the gain in the training dynamics (minima become saddles). The conjecture puts it in the expected loss of a random function. These are different claims. The paper gives no evidence for the second
- **What the idea can borrow:**
  - The split-neuron test as a control experiment. Compare two networks with equal `l(f)` at initialization and different parameter counts. If the loss after training is different, arc length is not the full explanation
  - The teacher-student task with 2D input and a clear success criterion. There the graph of `f` is a surface in 3D, so `l(f)` (surface area) and the distance loss `L(f, D)` can be computed exactly and compared with the success rate for each width
  - The strict test of an analogy: write the properties that the "squiggly line" picture implies (for example: more length -> lower loss, independent of the rest), derive a scaling law, and compare it with measured data, as the paper does with `p0^N`
  - For the "wind tunnel" picture of depth: Corollary 3.3 says that added neurons in any layer keep the old critical points as saddles. This is a result about width in each layer. It gives a possible way to state what a "longer path" adds

## Similar papers

- [On the Expressive Power of Deep Neural Networks](https://arxiv.org/abs/1606.05336)
- [Deep ReLU Networks Preserve Expected Length](https://arxiv.org/abs/2102.10492)
- [The Geometric Occam's Razor Implicit in Deep Learning](https://arxiv.org/abs/2111.15090)
- [A Universal Law of Robustness via Isoperimetry](https://arxiv.org/abs/2105.12806)
- [How many degrees of freedom do we need to train deep networks: a loss landscape perspective](https://arxiv.org/abs/2107.05802)
- [Loss Landscapes are All You Need](https://openreview.net/forum?id=QC10RmRbZy9)
- [What's Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/abs/1911.13299)
- [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555)
- [Large language models implicitly learn to straighten neural sentence trajectories](https://arxiv.org/abs/2311.04930)
- [Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
