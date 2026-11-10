# [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555)

## Meta

* Journal   - ICML
* Year      - 2024
* Author    - Rice University, Brown University (Humayun, Balestriero, Baraniuk)
* Code      - No code repository is given. Project page: [bit.ly/grok-adversarial](https://bit.ly/grok-adversarial)
* Slides    - 
* One liner - Grokking occurs in many practical settings. Local complexity (density of linear regions) shows a late phase where the regions leave the data points and go to the decision boundary.
* Model     - MLP, CNN, ResNet-18, GPT-style transformer
* Datasets  - MNIST, CIFAR10, CIFAR100, Imagenette, ImageNet, Shakespeare text, modular addition
* Baselines - None (analysis paper)

## What problem does it solve?

- Grokking = the network generalises a long time after it fits the training data
- Earlier work saw grokking only in special settings (algorithmic tasks, large initialisation, large weight decay)
- The paper says that grokking is much more common. It occurs in a CNN on CIFAR10 and a ResNet on Imagenette
- The paper adds a new type: **delayed robustness**. The network becomes robust to adversarial examples a long time after it fits the data and / or generalises
- Earlier explanations use measures that need the labels, the loss or the task. The paper wants one measure that needs none of these

## How does it solve it?

This is an analysis paper. It has no new training method.

### Setup

1. See a ReLU network as a continuous piecewise affine function (a spline). The input space is divided into linear regions. In each region the network is one affine map
2. Each neuron gives one hyperplane (a "non-linearity" or kink). The hyperplanes are the borders of the regions
3. Define **local complexity (LC)** = density of these non-linearities in a small neighbourhood of a point
4. During training, measure LC around training points, test points and random points
5. At the same steps, measure training accuracy, test accuracy and accuracy on adversarial examples (PGD on the test set)

How LC is approximated:

1. For an input `x`, sample `P` orthonormal directions `v_p`
2. Make a cross-polytope with vertices `x ± r · v_p`. Its convex hull is the neighbourhood
3. Send the vertices through the first layer. A neuron cuts the neighbourhood if its pre-activation has different signs at different vertices
4. Count these neurons
5. Send the vertices to the next layer and do the count again. Sum across layers

Defaults: `P = 25`, `r = 0.005` (`r = 1e-4` for ResNet18 on Imagenette). LC is measured at 1,024 points (MNIST: 1,000 training points, 10,000 test and random points).

### Main results

LC around the data has three phases. The paper names this a double descent of LC.

1. **First descent:** LC decreases immediately after initialisation. This phase depends on the initialisation and the parameterisation. It can be absent
2. **Ascent:** LC increases around training and test points. This usually continues until the network fits the training data
3. **Second descent (region migration):** the non-linearities move away from the training points, and also from test and random points. They collect near the decision boundary. The function becomes linear in a large area around each data point
4. Delayed generalisation and delayed robustness start after the second descent begins. The paper says the second descent always comes first

Circuit view:

- The set of active neurons in a region is one "circuit". Neighbouring regions differ by one neuron
- Thus LC is also the density of different circuits near a point
- Late in training, the number of different circuits decreases a lot, mostly in the deeper layers. This agrees with the "cleanup" phase of Nanda et al. (2023)

### Equations

#### Network as an affine spline

`f(x) = Σ_ω (A_ω x + b_ω) · 1[x in ω]`

- `ω` - one linear region of the input space
- `A_ω`, `b_ω` - slope and offset of the affine map in that region

#### Hyperplane of neuron i

`<w_i, x> + b_i = 0`

#### Local complexity (approximation)

```python
def local_complexity(net, x, P=25, r=0.005):
    V = orthonormal_directions(P, dim=x.size)
    verts = [x + r * v for v in V] + [x - r * v for v in V]   # cross-polytope
    count = 0
    for layer in net.layers:
        pre = [layer.pre_activation(v) for v in verts]
        # a neuron cuts the neighbourhood if its sign is not the same at all vertices
        count += number_of_neurons_with_sign_change(pre)
        verts = [layer.activation(p) for p in pre]             # go to the next layer
    return count
```

#### Adversarial examples

- `l_inf` PGD, `ε in {0.03, 0.06, 0.10, 0.13, 0.16, 0.20}`, step size `α = 0.0156`, 10 steps (100 steps for MNIST)

### Model

- MLP with ReLU on MNIST (main example: 4 layers, width 200, 1,000 training samples, 1e5 steps, batch size 200)
- CNN with 5 convolution layers and 2 linear layers on CIFAR10 and CIFAR100
- ResNet18 without batch norm on CIFAR10 (pre-activation, width 16)
- ResNet18 on Imagenette. ResNet18 with batch norm on ImageNet
- GPT-style transformer (12 layers, 12 heads) on Shakespeare text. PGD with `ε = 0.03` in the token-embedding space
- Transformer on modular addition. GeLU MLP
- Optimizer: Adam. Weight decay 0, but 0.01 for the MNIST MLP. No batch norm in the main experiments

## How is this paper novel?

- Shows grokking in common settings, not only in algorithmic tasks
- Gives the name and the evidence for delayed robustness
- Local complexity: a progress measure that does not use labels, loss or dataset
- Finds the three-phase dynamics of LC and the region migration
- Connects linear regions to circuits, and thus to mechanistic interpretability
- Shows that batch norm stops region migration and stops grokking

## List of experiments

- **ResNet18 on CIFAR10:** accuracy on adversarial examples starts to increase near 1e4 steps. Near 2e5 steps, robust accuracy is almost equal to test accuracy
- **MLP on MNIST, CNN on CIFAR10 / CIFAR100, ResNet18 on Imagenette, transformer on modular addition, GPT on Shakespeare:** all show the LC dynamics and delayed robustness
- **Auto-Attack in place of PGD:** delayed robustness is still there
- **GeLU in place of ReLU:** similar LC dynamics
- **Toy regression of `sin(x1) + cos(x2)` on one side of the input:** linear regions collect where the target has curvature

### Ablation Studies

Effect on grokking of adversarial examples:

- **Depth 2 -> 6 (MNIST MLP):** faster
- **Layers 6 -> 12 (Shakespeare GPT):** faster
- **Width 20 -> 2000 (MNIST MLP):** faster
- **Width 16 -> 64 filters (CIFAR10 ResNet18):** faster
- **Batch size 64 -> 512 (CIFAR10 ResNet18):** faster
- **Training data 1K -> 60K (MNIST MLP):** faster
- **Random labels, 1K -> 60K (MNIST MLP):** slower. Memorisation needs more capacity
- **Weight decay 0 -> 1 (MNIST MLP):** slower. The effect on LC is not monotonic
- **Add batch norm (CIFAR10 ResNet18 and CNN):** grokking does not occur or is much later. Batch norm removes the first descent, and LC continues to increase. Appendix B: batch norm moves the region borders towards the training data
- **Initialisation scale x8 (MNIST MLP):** causes grokking. No first descent. Robustness peaks come back periodically

More details:

- Width 20 to 1000: the ascent starts earlier and the peak LC is higher. Width 2000: the peak LC is lower. The authors read this as implicit regularisation
- Depth: on MNIST more depth makes the ascent peak less sharp. On CIFAR10 CNNs more depth increases the peak LC

Check of the LC approximation (Appendix A):

- Comparison with exact region counts from SplineCam (depth 3, width 200): the same double descent shape
- For large `r` and deep networks, the neighbourhood is deformed a lot in the deeper layers. For `r <= 0.014` the deformation stays low

### Efficiency analysis

- LC for 1,000 ImageNet test samples on ResNet18: about 28 s on an RTX 8000
- No other cost is given

## Preliminaries

### Grokking

The training accuracy goes to about 100% early. The test accuracy stays low for many more steps. Then the test accuracy increases suddenly.

### Linear regions of a ReLU network

- A ReLU neuron is linear on each side of its hyperplane
- All hyperplanes together divide the input space into regions
- In one region, the full network is one affine map `A x + b`
- More regions near a point = the function can bend more there

### PGD adversarial attack

Find a small change `δ` with `|δ|_inf <= ε` that makes the loss as large as possible.

```python
x_adv = x
for step in range(n_steps):
    x_adv = x_adv + alpha * sign(grad(loss(f(x_adv), y), x_adv))
    x_adv = clip(x_adv, x - eps, x + eps)
```

## GPU hours

Not mentioned. The only number is 28 s on an RTX 8000 for one LC measurement on 1,000 ImageNet samples.

## Key takeaways

- A network continues to change after it fits the training data. Longer training can give robustness
- First the network puts kinks near the training points (ascent). Then it moves them away, to the decision boundary (region migration)
- A network that generalises and is robust is locally linear around the data. It is non-linear only near the decision boundary
- Wider and deeper networks grok faster. More real data also makes it faster
- Batch norm keeps the kinks near the data and stops grokking
- The paper has no theory for why the second descent occurs. All experiments use Adam

## What I still do not understand?

## Ideas to pursue

- **In common with the Squiggly Line Conjecture:** the two use the same picture. A ReLU network is a piecewise linear graph, and the number of kinks near a point (local complexity) is a direct, measurable form of "squiggliness". The LC estimator is a tool to count the squiggles around data points, at initialisation and during training
- **Where it supports and where it contradicts:**
  - Contradiction: the conjecture says that a more squiggly curve near the data is better. The paper shows that generalisation and robustness come when the network removes the squiggles from the area around the data points and moves them to the decision boundary. The final good function is smooth (linear) near the data. More squiggle near the data is the memorisation phase, not the goal
  - Partial support: the ascent phase agrees with the conjecture. To fit the training points, the network increases the kink density around them, which is "the curve bends to go through the points"
  - Partial support for "larger networks learn faster": more width and more depth make grokking faster. But width 2000 has a lower peak LC than width 1000, so the larger network is not more squiggly near the data
  - The first descent says that the squiggle of the random initial function near the data is removed first. This is evidence against "squiggle at initialisation is the useful resource"
- **What the idea can borrow:**
  - Replace the global arc length `l(f)` with a local density of kinks. The paper shows that the position of the squiggles is more important than the total quantity
  - Use the LC estimator to test the conjecture at initialisation: measure LC around data points for random networks of different width and depth, and compare with `L(f, D)`
  - The "wind tunnel" picture of a classifier agrees with region migration: the turns of the flow are at the class borders, and the flow is straight elsewhere

## Similar papers

- [On the Expressive Power of Deep Neural Networks](https://arxiv.org/abs/1606.05336)
- [Deep ReLU Networks Preserve Expected Length](https://arxiv.org/abs/2102.10492)
- [The Geometric Occam's Razor Implicit in Deep Learning](https://arxiv.org/abs/2111.15090)
- [A Universal Law of Robustness via Isoperimetry](https://arxiv.org/abs/2105.12806)
- [How many degrees of freedom do we need to train deep networks: a loss landscape perspective](https://arxiv.org/abs/2107.05802)
- Loss Landscapes are All You Need (ICLR 2023)
- [What's Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/abs/1911.13299)
- [Large language models implicitly learn to straighten neural sentence trajectories](https://arxiv.org/abs/2311.04930)
- [Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
