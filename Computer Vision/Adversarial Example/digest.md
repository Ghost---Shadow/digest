# Adversarial Examples Digest

## [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199)

**Time to read:** 105 mins

**Easy to read:** Yes

**Author:** Christian Szegedy et al

**Year of Submission:** 2014

### What problem does it solve?

Interpretability of neural networks.

1. Semantic meaning of individual units.

2. Stability of neural networks with respect to small peturbations of its inputs.

### How does it solve it?

1. Maximum stimulation in a random basis.

2. Make small peturbations to the input image which are visually similar but increase the classification error. 

### How is this paper novel?

Flagship paper

### Key takeaways

1. There is no distinction between individual high level units and random linear combinations of high level units.

2. It is the space, rather than the individual units, that contains the semantic information in the high layers of neural networks.

3. Neural networks learn input-output mappings that are fairly discontinuous to a significant extent. 

4. **If we use one neural net to generate a set of adversarial examples, we find that these examples are still statistically hard for another neural network even when it was trained with different hyperparameters or, most surprisingly, when it was trained on a different set of examples. However the effectivenes decreases considerably.**

5. That assumption that local generalization i.e. the network is able to classify samples near its training input, does not hold. In other words there are adversarial pockets in the manifold.

6. Back feeding adversarial examples to training might improve generalization.

7. It is possible to generate examples that are readable by computers but not by humans.

8. Autoencoders are very resilient to adversarial examples but are not completely immune. 

9. A conservative measure of the unstability of the network can be obtained by simply computing the operator norm of each fully connected and convolutional layer. (Occam's Razor?)

### What I still do not understand?

1. **Formal description:** The exact function for generating the adversarial examples.

2. **Spectral Analysis of Unstability:** 
    1. Lipschtz constant.
    2. Contrast normalization layer.
    3. Parseval's formula.

### Ideas to pursue

1. **Hypothesis:** Only networks with convolutional filters fall victim to adversarial examples.

## [Practical Black-Box Attacks against Machine Learning](https://arxiv.org/abs/1602.02697)

**Time to read:** 105

**Easy to read:** No. (Heavy on advanced math)

**Author:** Nicolas Papernot et al

**Year of Submission:** 2017

### What problem does it solve?

Malicious inputs to self driving cars can cause undesirable behaviours. This paper demonstrates potential exploits where the attacker has no knowledge of the model nor can he train a substitute model.

### How does it solve it?

1. **Initial collection:** The adversary collects a very small set of inputs representative of the input domain.

2. **Architecture selection:** The adversary selects an architecture to be trained as the substitute F.

3. **Substitute training:** The adversary iteratively trains more accurate substitute DNNs by repeating the following.
    1. **Labeling:** Queries for lables from the oracle.
    2. **Training:** Adversary trains normally as well as using substitute training set
    3. **Augmentation:** The adversary applies an augmentation technique on the substitution set to produce a larger substitution set.

### How is this paper novel?

It provides a demonstration that black box attacks against DNN classifiers are practical for real world adversaries with no knowledge about the model or has access to any indepedently collected large training dataset. The adversary's only capability is to observe labels assigned by the DNN for chosen inputs in a manner analog to a cryptographic oracle.

### Key takeaways

1. The labels are not only misclassified by the substitute but also by the target DNN, because both models have similar decision boundaries.

2. Using gaussian noise to select points on which to train substitutes did not work likely due to noise not being representative of the input distribution.

3. The attack generalizes to non differentiable target oracles like decision trees.

4. Gradient masking can be used to defend against adversarial attacks. e.g. using kNN instead of using DNN. However substitute model overcomes that.

5. Training a model on adversarial examples makes the model more robust.

6. Defensive distillation reduces the gradients in local neighborhoods of training points. However, the substitute model is not distilled, and as such possesses the gradients required for the fast gradient sign method to be successful when computing adversarial examples.

7. Defending against finite perturbations is a more promising avenue for future work than defending against innitesimal perturbations.

### What I still do not understand?

1. x = ~x + arg minf~z : ~O(~x + ~z) 6= ~O(~x)g = ~x + ~x

2. **The heuristic used to generate synthetic training inputs is based on identifying directions in which the model's output is varying, around an initial set of training points. Such directions intuitively require more input-output pairs to capture the output variations of the target DNN O.**

3. **Therefore, to get a substitute DNN accurately approximating the oracle's decision boundaries, the heuristic prioritizes these samples when querying the oracle for labels. These directions are identied with the substitute DNN's Jacobian matrix JF, which is evaluated at several input points ~x.**

4. Goodfellow et al. algorithm: Fast gradient sign method

5. Papernot et al. algorithm

6. Reservoir sampling

7. Defensive distillation

8. For two "similar" architectures F and G distributions DF and DG induced by a population distribution D are highly correlated.

### Ideas to pursue

1. Adversarial attacks on models processing textual data?

2. **Hypothesis:** If two models with completely different weights or architectures perform equally in a task then there must be some lower dimensional projection of the architecture and/or weights which is equal. This way we can show that multiple different models are effectively the same model. We can then traverse this projected hyperplane to find a model which has similar behaviour but has a significantly smaller footprint.
