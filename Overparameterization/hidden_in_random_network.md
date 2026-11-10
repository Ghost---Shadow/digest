# [What's Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/abs/1911.13299)

## Meta

* Journal   - CVPR
* Year      - 2020
* Author    - Allen Institute for Artificial Intelligence, University of Washington (Ramanujan, Wortsman, Kembhavi, Farhadi, Rastegari)
* Code      - [allenai/hidden-networks](https://github.com/allenai/hidden-networks)
* Slides    - 
* One liner - A randomly weighted network contains a subnetwork that performs well with no weight training. The edge-popup algorithm finds it by learning a score for each weight.
* Model     - Conv2 to Conv8 (CIFAR-10), ResNet-50, ResNet-101, Wide ResNet-50 (ImageNet)
* Datasets  - CIFAR-10, ImageNet
* Baselines - Trained dense networks of the same architectures

## What problem does it solve?

- Usual training changes the weight values. The paper asks: is training of the weight values necessary at all?
- Claim: a large network with random, fixed weights already contains a subnetwork that has good accuracy
- The problem is then a search problem: find that subnetwork, and do not change a weight value
- The earlier method for this search (the "supermask" of Zhou et al.) uses stochastic sampling of the subnetwork. It does not scale well to ImageNet

## How does it solve it?

### Training flow

The algorithm is named `edge-popup`.

1. Initialise the network with random weights. The weights are fixed for all of the training
2. Give each weight `w_uv` one learnable score `s_uv`. Initialise the scores with Kaiming uniform
3. Forward pass: in each layer, keep only the weights with the top-k% scores. Set the other weights to zero for this pass
4. Calculate the loss of this subnetwork on the mini-batch
5. Backward pass: the top-k selection has no gradient. Use the straight-through estimator (treat the selection as the identity function)
6. Update the scores with SGD. Do not update the weights
7. An edge that is not in the subnetwork can get a higher score and "pop" back into the subnetwork. An edge in the subnetwork can go out
8. Batch norm parameters are also fixed (bias 0, scale 1)

For convolution layers, the top-k% is taken across all kernel weights of the layer. The gradient is summed across all spatial positions.

### Inference

1. Take the fixed random weights
2. Apply the final top-k% mask from the scores
3. Use the masked network as a normal network

### Equations

#### Forward pass

`I_v = Σ_u w_uv · Z_u · h(s_uv)`

- `I_v` - input of node v
- `Z_u` - output of node u, `Z_v = σ(I_v)`
- `h(s_uv)` - 1 if `s_uv` is in the top-k% scores of the layer, 0 if not

#### Score gradient (straight-through estimator)

`g(s_uv) = (∂L/∂I_v) · w_uv · Z_u`

#### Score update (SGD, momentum and weight decay ignored)

`s_uv = s_uv − α · (∂L/∂I_v) · w_uv · Z_u`

```python
# one layer, python-like pseudocode
mask = top_k_mask(abs(scores), k)          # 1 for top-k% scores, 0 for the others
out = layer(x, weights * mask)             # weights are fixed
loss.backward()                            # gradient goes "straight through" the mask
scores -= lr * scores.grad                 # only the scores change
```

#### Theorem 1 (a swap decreases the loss)

- When edge `(i, ρ)` replaces edge `(j, ρ)` and the other edges stay the same, the loss on the mini-batch decreases
- Condition: the loss is sufficiently smooth and the change of the node input is small
- Proof idea: first-order Taylor expansion of the loss. The swap occurs only when the score of the new edge became larger than the score of the old edge, and this makes the first-order change negative
- The general version covers more than one swap in a layer

#### Intuition for "larger is better"

`P(no good subnetwork) ≈ (1 − q)^S`

- `q` - probability that one subnetwork has weights near enough to a trained target network
- `S` - number of subnetworks. `S` becomes very large when the width increases
- Thus the probability of "no good subnetwork" goes to zero for a very wide network
- The number of subnetworks `C(n, k·n)` is largest near `k = 0.5`

### Model

- **CIFAR-10:** VGG-style Conv2, Conv4, Conv6, Conv8. All have fully connected layers 256, 256, 10
- **ImageNet:** ResNet-50, ResNet-101, Wide ResNet-50
- **Weight distributions:**
  - Kaiming Normal: `std = sqrt(2 / fan_in)`
  - Signed Kaiming Constant: each weight is `+std` or `-std` with a random sign
  - Scaled variants: multiply the std by `sqrt(1/k)` to compensate for the removed weights
- **CIFAR-10 training of scores:** SGD, learning rate 0.1, cosine decay, momentum 0.9, weight decay 1e-4, batch size 128, 100 epochs, 5 seeds
- **ImageNet training of scores:** hyperparameters of the public NVIDIA ResNet example. The paper does not list them

## How is this paper novel?

- Shows that a random network contains a subnetwork with good accuracy, with no training of the weight values, at ImageNet scale
- `edge-popup`: a deterministic top-k% selection on scores, not a stochastic sample as in Zhou et al.
- Gives a proof that each swap of edges decreases the mini-batch loss (first order)
- Shows that the weight distribution is important: random signs with one constant magnitude are better than Kaiming Normal

## List of experiments

- **ImageNet, edge-popup at k = 30% (top-1)**
  - Trained baselines: ResNet-34 (21.8M parameters) 73.3, ResNet-50 (25M) 76.1, Wide ResNet-50 (69M) 78.1
  - Kaiming Normal: ResNet-50 (7.6M weights kept) 61.71, ResNet-101 (13M) 66.15, Wide ResNet-50 (20.6M) 67.95
  - Signed Kaiming Constant: ResNet-50 68.6, ResNet-101 72.3, Wide ResNet-50 73.3
  - Headline: a subnetwork of a random Wide ResNet-50 (20.6M weights) has the same accuracy as a trained ResNet-34 (21.8M weights)
- **CIFAR-10, Conv2 to Conv8**
  - Deeper and wider random networks give better subnetworks
  - With sufficient width, a 50% subnetwork of random Conv6 has the same accuracy as the trained dense Conv6
  - Exact numbers are only in the figures
- **Comparison with Zhou et al.**
  - edge-popup is much better on CIFAR-10 and on ImageNet
  - Zhou et al. report 65.4% on CIFAR-10 in their own paper

### Ablation Studies

- **Effect of k:** accuracy is best for k in [30, 70]. It is worst near k = 0 and k = 100. k = 100 is the full random network with no selection. On ImageNet, k = 30 was best
- **Effect of width:** a wider network decreases the gap to the trained dense network. The gain stays when the number of kept parameters is held constant, so the larger search space is the cause
- **Effect of weight distribution:** Signed Kaiming Constant is better than Kaiming Normal. Xavier normal and Kaiming uniform are much worse on CIFAR-10. The scaled variant helps on ImageNet
- **Train the found subnetwork:** accuracy is not substantially better than that of a trained random subnetwork (Wide ResNet-50, Signed Kaiming Constant: 73.3 before weight training, 76.9 after). The paper says that the found subnetworks are thus not lottery tickets in the sense of Frankle and Carbin

### Efficiency analysis

Not mentioned

## Preliminaries

### Lottery ticket hypothesis

A dense network at initialisation contains a sparse subnetwork (the "winning ticket"). When you train this subnetwork alone from the same initial weights, it gets the accuracy of the full network.

- Lottery ticket: find the subnetwork, then train its weights
- This paper: find the subnetwork, and do not train the weights

### Straight-through estimator

A function such as `top_k` or `sign` has a gradient of zero almost everywhere.

- Forward pass: use the real function
- Backward pass: use the gradient of the identity function as a replacement

```python
forward:  y = h(s)
backward: dy/ds = 1
```

### Kaiming initialisation

Weights come from a distribution with `std = sqrt(2 / fan_in)`. This keeps the variance of the activations constant across ReLU layers.

## GPU hours

Not mentioned

## Key takeaways

- The weight values do not have to be learned. The selection of connections is sufficient for good accuracy
- A full random network (k = 100%) is bad. The good accuracy comes from the selection
- A larger random network contains better subnetworks, because it has more candidate subnetworks
- The sign pattern is the important part of a random weight. The magnitude can be one constant
- The search for the subnetwork costs about as much as usual training (one score per weight, same forward and backward pass)

## What I still do not understand?

## Ideas to pursue

- **In common with the Squiggly Line Conjecture:** both say that a larger random network is better before any weight is trained. The paper's argument `(1 − q)^S` is a coverage argument: more candidate subnetworks means a higher probability that one of them is near the target. This is the same shape of argument as "a longer curve covers more of the space". The paper also gives a direct link to the lottery ticket hypothesis, as the conjecture claims
- **Where it supports and where it contradicts:**
  - Support: with the number of kept weights held constant, a wider random network still gives a better subnetwork. Size at initialisation helps
  - Contradiction: the conjecture is about the expected loss of the full random function. The paper shows the opposite for the full function: k = 100% (no selection) is the worst case. The benefit of size is in the best subnetwork (a minimum across `S` candidates), not in the average random function. The conjecture must change from "lower expected L" to "lower minimum L across subnetworks", or it does not agree with this paper
  - The paper measures classification accuracy after a search, not a distance from data points to the graph of `f`. It gives no data about arc length
- **What the idea can borrow:**
  - Use `S` (number of subnetworks) as the measure of "coverage", not only arc length. Diminishing returns then come from `(1 − q)^S`
  - Measure the arc length and `L(f, D)` of edge-popup subnetworks for different widths and different k. This is a direct test of the conjecture with public code
  - The result on Signed Kaiming Constant says that the sign pattern decides the function. For a ReLU network the sign pattern decides where the kinks of the piecewise linear graph are, and this is the "squiggle" in the conjecture

## Similar papers

- [On the Expressive Power of Deep Neural Networks](https://arxiv.org/abs/1606.05336)
- [Deep ReLU Networks Preserve Expected Length](https://arxiv.org/abs/2102.10492)
- [The Geometric Occam's Razor Implicit in Deep Learning](https://arxiv.org/abs/2111.15090)
- [A Universal Law of Robustness via Isoperimetry](https://arxiv.org/abs/2105.12806)
- [How many degrees of freedom do we need to train deep networks: a loss landscape perspective](https://arxiv.org/abs/2107.05802)
- Loss Landscapes are All You Need (ICLR 2023)
- [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555)
- [Large language models implicitly learn to straighten neural sentence trajectories](https://arxiv.org/abs/2311.04930)
- [Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
