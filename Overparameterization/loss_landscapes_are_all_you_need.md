# [Loss Landscapes are All You Need: Neural Network Generalization Can Be Explained Without the Implicit Bias of Gradient Descent](https://openreview.net/forum?id=QC10RmRbZy9)

## Meta

* Journal   - ICLR (oral)
* Year      - 2023
* Author    - University of Maryland, New York University, Max Planck Institute for Software Systems (Chiang, Ni, Miller, Bansal, Geiping, Goldblum, Goldstein)
* Code      - [Ping-C/optimizer](https://github.com/Ping-C/optimizer)
* Slides    - 
* One liner - Gradient-free optimisers, including random guess-and-check, generalise about as well as SGD in low-data settings. Good minima have more volume, so most procedures that fit the data find them.
* Model     - One-hidden-layer network (toy), LeNet
* Datasets  - MNIST, CIFAR-10, CIFAR-FS, mini-ImageNet, toy 2D "wedge" data, slab data (Shah et al.)
* Baselines - SGD, against Guess & Check and Pattern Search

## What problem does it solve?

- Overparametrized networks can fit random labels, but they generalize well. Classical theory says that they must overfit
- The usual explanation: gradient descent (SGD) has an implicit bias. It selects good minima from all the minima that fit the data
- The paper tests if gradient dynamics are necessary for this bias
- Answer of the paper: no, at least in the low-data regime. Optimizers that use no gradient generalize almost as well as SGD
- Alternative explanation, the **volume hypothesis**: in weight space, the minima that generalize well have a much larger total volume than the minima that generalize badly. Any optimizer finds the large-volume set

## How does it solve it?

### Training flow

Guess & Check (G&C)

1. Sample all weights independently and uniformly from `[-1, 1]`
2. Compute the training accuracy and the training loss
3. If the accuracy is 100% and the loss is below a selected threshold, keep the model and stop
4. If not, discard the model and go to step 1
5. No gradient and no iterative steps. The probability to select a set of solutions is exactly proportional to its volume in weight space

Pattern Search

1. Select a random parameter
2. Move it by a fixed step. The sign is random
3. If the loss decreases, keep the new point
4. If no parameter gives a decrease, divide the step size by 2 (start radius 1)
5. Stop at 100% training accuracy

Random Greedy Search

1. Add Gaussian noise with standard deviation `sigma` to all weights (start `sigma = 1`)
2. If the training loss decreases, keep the new point
3. If 30000 random steps give no decrease, divide `sigma` by 2
4. Stop at 100% training accuracy

Comparison with SGD

1. Normalize the weights of the model, then compute the train loss
2. Put each model in a bin of train loss (for example `(0.3, 0.35)`)
3. Compare the test accuracy of G&C and SGD in the same bin and with the same number of samples. This is necessary because a lower train loss gives better generalization, also after 100% train accuracy

### Inference

1. No change. The model that the optimizer returns is used as a normal classifier

### Equations

#### Guess & Check

```python
def guess_and_check(model, X, y, loss_threshold):
    n_guesses = 0
    while True:
        w = uniform(-1, 1, size=model.num_params)
        n_guesses += 1
        if accuracy(model(w, X), y) == 1.0 and loss(model(w, X), y) < loss_threshold:
            return w, n_guesses
```

#### Volume estimate

`volume(solution set) ≈ 1 / (number of guesses until a solution is found)`

- Slab data: the linear decision boundary has volume about `10^-4`. The robust (more complex) decision boundary has volume below `10^-10`

#### Cost

- The number of guesses increases exponentially with the number of training samples and the number of classes
- For this reason G&C is used only on 2-class problems with a maximum of 32 samples

### Model

- Toy problem: a neural network with one hidden layer of 2, 4, 10, 15 or 20 hidden units
- LeNet for MNIST and CIFAR-10. The MNIST LeNet has 11074 parameters
- Linear model on MNIST as a comparison with a much smaller hypothesis class
- LeNet with a width factor from 0.2 to 1 for the scaling experiment
- ResNet-12 backbone for few-shot learning. The feature extractor is pre-trained with gradients. Only the new classification head is trained with Pattern Search or SGD

## How is this paper novel?

- It removes the optimizer from the question. G&C has no dynamics, so its only bias is the geometry (volume) of the loss landscape
- It shows by direct sampling that the *total* volume of good minima is large. Earlier work (Huang et al.) showed only that single good minima are larger than single bad minima
- It shows that three gradient-free optimizers generalize like SGD on MNIST, CIFAR-10 and few-shot tasks
- It shows that the gain from a wider network is also there with G&C, without SGD
- It gives a volume explanation for the simplicity bias that was attributed to SGD

## List of experiments

- **Toy "wedge" problem (11 train points, 5 test points)**
  - A "poisoned" network (trained to fit the train set and to fail on the test set) shows that bad decision boundaries exist. They become more irregular from 2 to 20 hidden units
  - SGD gives almost the same simple boundary for all widths
  - G&C also gives simple boundaries for all widths
- **Poisoned LeNet (Table 1):** LeNet can fit 2 to 32 MNIST samples (2 to 24 CIFAR samples) and get 0% validation accuracy. So bad minima exist in these models
- **2-class MNIST, best test accuracy (Table 2), 175 random splits and seeds**
  - 32 samples: G&C 93.02, SGD 94.04, linear model 84.75
  - 16 samples: G&C 89.21, SGD 91.24, linear model 80.68
  - 8 samples: G&C 83.05, SGD 84.82, linear model 74.29
  - 4 samples: G&C 76.28, SGD 77.35, linear model 65.12
  - 2 samples: G&C 66.89, SGD 69.67, linear model 58.93
- **2-class CIFAR-10, best test accuracy (Table 3), 75 random splits and seeds**
  - 24 samples: G&C 66.59, SGD 63.16
  - 16 samples: G&C 61.10, SGD 58.98
  - 8 samples: G&C 57.17, SGD 56.76
  - 4 samples: G&C 55.51, SGD 53.75
  - 2 samples: G&C 52.39, SGD 51.98
- **10-class, more samples (Table 4), SGD / Pattern Search / Random Greedy**
  - MNIST 1000 samples: 93.46 / 93.68 / 93.34
  - MNIST 100 samples: 78.67 / 78.43 / 78.51
  - CIFAR-10 1000 samples: 36.01 / no result (time limit) / 34.44
  - CIFAR-10 100 samples: 19.86 / 18.70 / 16.80
  - Mean difference to SGD: 0.9%
- **Few-shot, 1-shot 5-way, ResNet-12, 600 episodes (Table 5)**
  - CIFAR-FS: SGD 68.35, Pattern Search 70.25
  - mini-ImageNet: SGD 55.76, Pattern Search 58.53
- **Slab data (simplicity bias):** volume of the linear boundary about `10^-4`, volume of the robust boundary below `10^-10`

### Ablation Studies

- **Width of LeNet with G&C (Table 6, 2-class MNIST, 16 samples):** the test accuracy increases with the width. Example for the loss bin `(0.3, 0.35)`: width 0.2 -> 77.39, width 0.4 -> 85.82, width 0.6 -> 91.20. For large widths many cells have no result
- **Sampling range (Tables 7 and 8):** uniform `(-1, 1)`, `(-2, 2)`, `(-5, 5)` and a sphere with L2 norm 100 give almost the same accuracy
- **Train loss level:** in all tables, a lower train loss bin gives a higher test accuracy for G&C and for SGD
- **Linear model vs. LeNet:** the linear model has a much smaller hypothesis class, but it is about 10% worse than G&C on LeNet
- **Number of samples on the toy problem (32 and 64 samples):** the boundaries of SGD and G&C stay simple

### Efficiency analysis

- G&C has exponential cost in the number of samples and classes. Some cells are empty because no model was found in a reasonable time
- Pattern Search on CIFAR-10 with 1000 samples timed out
- No times or hardware numbers are given

## Preliminaries

### Implicit bias of gradient descent

- An overparametrized network has many weight settings with zero training error. Some generalize, some do not
- No explicit regularizer tells the optimizer which one to select
- The common hypothesis: the dynamics of SGD prefer some solutions (low rank, low norm, flat). This preference is the implicit bias

### Zeroth-order optimization

- An optimizer that uses only the value of the loss, not its gradient
- In the worst case, the number of loss evaluations increases exponentially with the dimension. This is the reason why gradients are normally used

### Volume vs. flatness

- Flat minimum: one wide basin
- Large-volume set: possibly many small regions that have a large volume together
- The paper speaks about volume. It does not claim that the good set is one flat basin

## GPU hours

Not mentioned

## Key takeaways

- A model sampled at random from all models that fit the data generalizes almost as well as a model trained with SGD (low-data regime)
- So the main bias is in the map from weights to functions (the loss landscape), not in the optimizer
- Bad minima exist, but their volume is very small
- Wider networks are better also with random sampling. Width increases the volume of the good functions
- SGD and G&C are not identical. SGD can have a small additional bias. The paper says the gradient bias is secondary, not zero
- Limits: only small sample counts, only small models for G&C, and no direct proof that SGD follows volume

## What I still do not understand?

## Ideas to pursue

- **In common with the Squiggly Line Conjecture:** this is the nearest paper to the idea. Both look at *randomly initialized* functions, with no training, and both say that the gain from more parameters is already there in the random functions. Table 6 is direct evidence: with pure random sampling, a wider LeNet gives a better model. The weights come from `[-1, 1]^N`, the same type of box as in the conjecture
- **Where it contradicts:** the mechanism is opposite. The conjecture says that the larger network is better because its random functions are *more squiggly* and cover more space. This paper shows that the random functions that fit the data are the *simple* ones: the decision boundaries of G&C stay simple from 2 to 20 hidden units, and the linear boundary has `10^6` times more volume than the complex one. More parameters add volume to the simple functions. They do not make the selected function longer. Also, the cost of G&C increases exponentially with the number of points. So a random curve does not get near to many data points cheaply, with any width
- **What the idea can borrow:**
  - The G&C protocol as a direct test of the conjecture. Sample random networks at some widths, compute `L(f, D)` (distance to the graph) and the arc length `l(f)`, and check if `l(f)` or the width predicts the loss
  - The volume estimate `1 / (number of guesses)`. Use it to measure how the probability of "the graph is within `eps` of all `n` points" changes with width and with `n`
  - The loss-bin comparison. Compare functions only at the same loss level, or the effect of width mixes with the effect of loss
  - A possible new form of the conjecture: replace "arc length of one function" with "volume of weight space that maps to functions near the data"

## Similar papers

- [On the Expressive Power of Deep Neural Networks](https://arxiv.org/abs/1606.05336)
- [Deep ReLU Networks Preserve Expected Length](https://arxiv.org/abs/2102.10492)
- [The Geometric Occam's Razor Implicit in Deep Learning](https://arxiv.org/abs/2111.15090)
- [A Universal Law of Robustness via Isoperimetry](https://arxiv.org/abs/2105.12806)
- [How many degrees of freedom do we need to train deep networks: a loss landscape perspective](https://arxiv.org/abs/2107.05802)
- [What's Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/abs/1911.13299)
- [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555)
- [Large language models implicitly learn to straighten neural sentence trajectories](https://arxiv.org/abs/2311.04930)
- [Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
