# Adversarial Examples Digest

## Intriguing properties of neural networks (Christian Szegedy et al)

Time to read: 105 mins
Easy to read: Yes

### What problem does it solve?

Interpretability of neural networks.

* Semantic meaning of individual units.

* Stability of neural networks with respect to small peturbations of its inputs.

### How does it solve it?

* Maximum stimulation in a random basis.

* Make small peturbations to the input image which are visually similar but increase. 

### How is this paper novel?

Flagship paper

### Key takeaways

* There is no distinction between individual high level units and random linear combinations of high level units.

* It is the space, rather than the individual units, that contains the semantic information in the high layers of neural networks.

* Neural networks learn input-output mappings that are fairly discontinuous to a significant extent. 

* **If we use one neural net to generate a set of adversarial examples, we find that these examples are still statistically hard for another neural network even when it was trained with different hyperparameters or, most surprisingly, when it was trained on a different set of examples. However the effectivenes decreases considerably.**

* That assumption that local generalization i.e. the network is able to classify samples near its training input, does not hold. In other words there are adversarial pockets in the manifold.

* Back feeding adversarial examples to training might improve generalization.

* It is possible to generate examples that are readable by computers but not by humans.

* Autoencoders are very resilient to adversarial examples but are not completely immune. 

* A conservative measure of the unstability of the network can be obtained by simply computing the operator norm of each fully connected and convolutional layer. (Occam's Razor?)

### What I still do not understand?

* **Formal description:** The exact function for generating the adversarial examples.

* **Spectral Analysis of Unstability:** 
1. Lipschtz constant.
2. Contrast normalization layer.
3. Parseval's formula.

### Ideas to pursue

* **Hypothesis:** Only networks with convolutional filters fall victim to adversarial examples.
