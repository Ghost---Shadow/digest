# [On the Expressive Power of Deep Neural Networks](https://arxiv.org/abs/1606.05336)

## Meta

* Journal   - ICML
* Year      - 2016 (arXiv v1), accepted to ICML 2017
* Author    - Cornell University, Google Brain, Stanford University (Raghu, Poole, Kleinberg, Ganguli, Sohl-Dickstein)
* Code      - Not mentioned
* Slides    - 
* One liner - Trajectory length, the length of the image of a 1-D input path, grows exponentially with depth for random networks with large weights, and other expressivity measures follow it.
* Model     - Random fully connected ReLU / hard tanh / tanh networks; small networks on MNIST and CIFAR-10
* Datasets  - MNIST, CIFAR-10
* Baselines - Batch norm (against trajectory regularization)

## What problem does it solve?

- How do depth and width change the set of functions that a network can compute?
- Older depth-separation results use hand-made, worst-case weights. It is not clear that a trained network looks like these functions
- Lower bounds had no matching upper bounds
- There was no link between expressivity measures and the performance that we see in practice
- The paper wants results that hold for random weights (at initialization) and that are also measured during training

## How does it solve it?

### Setup

1. Take a fully connected network with `n` hidden layers of width `k`, input in `R^m`, ReLU or hard tanh
2. Initialise weights `W ~ N(0, sigma_w^2 / k)` and biases `b ~ N(0, sigma_b^2)`
3. Take two input points `x0`, `x1` and a 1-D curve `x(t)`, `t in [0, 1]`, between them (a line, or a circular arc)
4. Send the curve through the network. `z_d(t)` is the image of the curve at layer `d`
5. Measure expressivity along the curve:
   - **Transitions** `T`: how many times a neuron changes its linear region as `t` goes from 0 to 1 (ReLU: off <-> on; hard tanh: -1 / linear / +1)
   - **Activation patterns** `A`: how many different on/off strings of all neurons occur along the curve
   - **Dichotomies**: sweep the weights of one layer along a curve (not the input) and count the different binary labelings of a fixed set of inputs
   - **Trajectory length**: arc length of `z_d(t)`
6. Show that the first three measures are controlled by the fourth one (trajectory length)

### Main results

1. **Theorem 1 (upper bound on activation patterns):** over all weight settings, the number of activation patterns is `O(k^(m*n))` for ReLU and `O((2k)^(m*n))` for hard tanh. The paper says the bound is asymptotically tight (it matches the construction of Montufar et al. 2014)
   - With a fixed budget of `N` neurons, the upper bound grows when the neurons are put into more layers: `U(1, N, m) < U(2, N/2, m) < ... < U(n, k, m)`
2. **Theorem 2 (regions):** a ReLU or hard tanh network cuts the input space into convex polytopes. The network is a different linear function on each polytope
3. **Theorem 3 (growth of trajectory length):** for a random network and a curve that is not a line, the expected length of the image grows exponentially with depth `d`. Width `k` is only in the base
4. **Theorem 4 (transitions are proportional to length):** for hard tanh with large initialization scale (neurons are saturated most of the time), the number of transitions grows at the same rate as the trajectory length
5. **Consequence 1, "all weights are not equal":** a perturbation at one layer grows exponentially in the remaining depth after that layer. Thus the lower (early) layers are more sensitive
6. **Consequence 2, trajectory regularization:** batch normalization makes trajectory length smaller. A direct penalty on trajectory length gives the same performance as batch norm

### Equations

#### Curves

```python
line(t)   = t * x1 + (1 - t) * x0
circle(t) = cos(pi * t / 2) * x0 + sin(pi * t / 2) * x1
```

#### Trajectory length

`l(x(t)) = integral over t of || dx(t)/dt || dt`

```python
# discrete version
length = sum(norm(z[i + 1] - z[i]) for i in range(len(z) - 1))
```

#### Theorem 3

ReLU:

`E[ l(z_d(t)) ] >= O( sigma_w * sqrt(k) / sqrt(k + 1) )^d * l(x(t))`

Hard tanh:

`E[ l(z_d(t)) ] >= O( sigma_w * sqrt(k) / sqrt( sigma_w^2 + sigma_b^2 + k * sqrt(sigma_w^2 + sigma_b^2) ) )^d * l(x(t))`

- `d` - depth of the layer
- `k` - width
- `sigma_w`, `sigma_b` - scale of weights and biases at initialization
- Condition: `x(t + delta)` has a non-zero component perpendicular to `x(t)` for all `t`, `delta` (the curve is not a line)
- The paper says the bound is tight in the limit of large `sigma_w` and `k`
- Note from the later paper of Hanin, Jeong, Rolnick (2102.10492): a factor 2 is missing in the ReLU statement. The correct base is `sigma_w * sqrt(k) / (2 * sqrt(k + 1))`

#### Theorem 4

`T = O( sqrt(k) / sqrt(1 + sigma_b^2 / sigma_w^2) )^n`

- `n` - number of hidden layers

#### Proof sketch of Theorem 3

1. Split the weight matrix: `W = W_parallel + W_perp` relative to `z_d`. For Gaussian `W` the two parts are independent
2. `W_parallel` sets the pre-activation, and thus which units are not saturated
3. Split the small step `delta_z` into a part parallel to `z_d` and a part perpendicular to `z_d`
4. Lower-bound the perpendicular part of `delta_z` at layer `d + 1` by the perpendicular part at layer `d` that goes through the unsaturated units
5. Apply the recurrence for each layer. ReLU uses `p = 1/2` for "unit is on"

#### Trajectory regularization

```python
# replaces each batch norm layer
loss += 0.01 * lam * (current_length / original_length)   # lam is learned
out = lam * activations
# length = piecewise linear path between adjacent datapoints of the minibatch
```

### Model

- Random fully connected ReLU / hard tanh / tanh networks (for example width 100) for the measurements at initialization
- Fully connected hard tanh network on MNIST, width 100, `sigma_w^2 = 1` (train one layer only). The same test on CIFAR-10 with width 200
- Convolutional ReLU network on CIFAR-10, `sigma_w^2 = 2`, with and without batch norm. The full architecture is not given (pooling at layers 3 and 7, then a fully connected layer)
- Optimizer, learning rate, batch size, epochs: not mentioned

## How is this paper novel?

- One quantity, trajectory length, explains three different expressivity measures
- The results are for random networks, not for one hand-made network. Thus they describe the typical case
- Gives an upper bound on activation patterns that is tight
- Connects the theory to practice: sensitivity of early layers, and a regularizer that can replace batch norm
- Measures trajectory length during training, not only at initialization

## List of experiments

- **Transitions vs. depth and width:** circular trajectory between MNIST datapoints. Transitions grow exponentially with depth. Width has a much weaker effect. The initialization scale matters
- **Trajectory growth by layer:** tanh network of width 100 (2-D projection of the curve for each layer); MNIST fully connected ReLU network; CIFAR-10 conv network. Conv layers also increase the length. Pooling layers decrease the length
- **Simulation vs. Theorem 3:** random hard tanh network, compared with the bound
- **Transitions vs. length:** for hard tanh, the number of transitions is approximately linear in trajectory length for different widths and weight variances
- **Noise in one layer (CIFAR-10 conv net):** noise of magnitude 0.25 in the first layer decreases accuracy by about 0.7. The same noise in the fifth layer decreases accuracy by about 0.02
- **Train one layer only (MNIST, all other layers frozen at random values):** training a lower hidden layer gives better performance than training a higher layer. CIFAR-10 shows almost the same trend, with lower accuracy
- **Length during training (CIFAR-10 conv net, no batch norm, `sigma_w^2 = 2`):** at step 0 the network is not in the exponential growth regime. Training increases the trajectory length
- **Weight norms during training:** weight scale increases during training. This pushes a network with a typical initialization into the exponential growth regime
- **Batch norm:** batch norm layers decrease trajectory length
- **Trajectory regularization vs. batch norm (CIFAR-10):** the two have comparable performance, and both are better than no batch norm. Only a qualitative result is given
- **Dichotomies:** hard tanh, 15 random inputs, `sigma_w^2 = 8` or `16`. Same trends for depth and width. The count depends on the number of layers above the swept layer, not on total depth

### Ablation Studies

- Initialization scale `sigma_w`: larger scale gives faster growth of length and transitions
- Width: changes the base of the exponent only
- Layer position: perturbation / training of lower layers has a larger effect than for upper layers
- With batch norm vs. without batch norm vs. trajectory regularization

### Efficiency analysis

- Trajectory regularization: training time is slightly longer than with batch norm. It has the same behaviour at train time and test time
- No other analysis is given

## Preliminaries

### Arc length

Length of a curve `x(t)`: add the lengths of very small steps.

`l = integral || dx/dt || dt`

```python
length = sum(norm(x[i + 1] - x[i]) for i in range(N - 1))
```

### Linear regions of a piecewise linear network

- ReLU is `max(0, x)`. Each neuron is "on" or "off"
- For a fixed on/off pattern of all neurons, the network is one linear function
- Thus the input space is cut into polytopes. One linear function for each polytope
- More regions = the function can bend more times = more expressive

### Hard tanh

```python
hard_tanh = lambda x: max(-1, min(1, x))
```

- Three regions: saturated at -1, linear, saturated at +1

## GPU hours

Not mentioned

## Key takeaways

- In a random network with large weight scale, the image of a curve becomes exponentially longer with depth
- Depth is the exponent. Width is only in the base, and the base factor `sqrt(k) / sqrt(k + 1)` goes to 1 for large `k`
- The growth rate depends on `sigma_w`. A network with a typical initialization (`sigma_w^2 = 2`) was not in the exponential regime at step 0. Training moves it into this regime
- The number of linear pieces along the curve follows the trajectory length
- Early layers are the most sensitive, because a change there is amplified by all the layers after it
- Batch norm decreases trajectory length. A penalty on trajectory length does the same job
- The theory is for training error / expressivity, not for generalization

## What I still do not understand?

## Ideas to pursue

- **In common with the Squiggly Line Conjecture:** this paper also sees a network as something that turns a simple curve into a longer, "more squiggly" curve, and uses arc length as the measure of what the network can express. Theorem 4 (number of linear pieces is proportional to length) agrees with the "ReLU networks are piecewise linear, not polynomials" intuition. The layer-by-layer growth of the curve is the "wind tunnel" picture: each layer bends the path again, and deeper means a longer path
- **Where it supports and where it contradicts:**
  - Supports: for random networks with a large weight scale, more depth gives an exponentially longer curve. The paper also shows that trained networks have longer trajectories than networks at initialization
  - Contradicts the "more parameters" form of the conjecture: the length is set by depth and by `sigma_w`, not by the parameter count. More width gives almost no extra length (base factor `sqrt(k) / sqrt(k + 1)`). A wide shallow network with many parameters is not squiggly
  - Contradicts at standard initialization: the paper's own CIFAR-10 network with `sigma_w^2 = 2` is not in the exponential regime at step 0. Hanin et al. (2102.10492) prove that with this scale the expected length does not grow with depth
  - Different object: the paper measures the image of a 1-D input curve in hidden / output space. The conjecture uses the graph of `f` over `[-1, 1]^d`, which is a `d`-dimensional surface, not a curve. The paper gives no statement about coverage of space or distance from data points to the curve
  - Batch norm and trajectory regularization make the curve shorter and performance better. This is against "longer is better" as a general rule
- **What the idea can borrow:**
  - The measurement recipe: send a line or circle between two datapoints through the network, measure length and count transitions, at initialization and during training
  - State the conjecture with depth and `sigma_w` as the variables, not the raw parameter count, and test width and depth separately
  - The "remaining depth" result for the autoregression picture: a small change early in the chain of hidden states is amplified by all later steps

## Similar papers

- [Deep ReLU Networks Preserve Expected Length](https://arxiv.org/abs/2102.10492)
- [The Geometric Occam's Razor Implicit in Deep Learning](https://arxiv.org/abs/2111.15090)
- [A Universal Law of Robustness via Isoperimetry](https://arxiv.org/abs/2105.12806)
- [How many degrees of freedom do we need to train deep networks: a loss landscape perspective](https://arxiv.org/abs/2107.05802)
- [Loss Landscapes are All You Need (ICLR 2023)](https://openreview.net/forum?id=QC10RmRbZy9)
- [What's Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/abs/1911.13299)
- [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555)
- [Large language models implicitly learn to straighten neural sentence trajectories](https://arxiv.org/abs/2311.04930)
- [Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
