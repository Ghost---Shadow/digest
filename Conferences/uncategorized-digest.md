# Talks and Conferences

## [A Neural Network Model That Can Reason - Prof. Christopher Manning](https://www.youtube.com/watch?v=24AX4qJ7Tts)

**Video length** 46:28

**Easy to follow?** Yes

**Author:** (Stanford) Chrispopher Manning

**Link to paper:** https://arxiv.org/abs/1803.03067

**Link to code:** https://github.com/stanfordnlp/mac-network

**Year of Upload:** 2018

### What problem does it solve?

Finding models that can exibit reasoning.

### How does it solve it?

#### Dataset

CLEVR

#### Models

1. **Partially differentiable model** - Translates the natural language question to a program which generates a model on the fly and then trains it.

2. **Relation net** - See `/Computer Vision/AbstractVisualReasoning/digest.md`

3. **FILM** - Inserts conditional linear normalization layers that tilt the activations based on the question.

4. **Memory. Attention. Composition.** - Sequence model where each cell does these actions. Exibits self attention. Memory is retrieved information relavant to a query. Composition is attention based average of a given query.

The query is processed by biLSTM. Concats final forward and backward hidden states.

Image features are extracted using a convnet.

Units inside a MAC cell

* Control unit: Computes a control state, extracting instruction that focuses on some aspect of the query.

* Read Unit: Retrieves from the knowledge base given the current control state and previous memory.

* Write Unit: Updates the memory state by merging old and new memory.

### How is this paper novel?

### Key takeaways

1. **Reasoning:** Algebraically manipulating previously acquired knowledge in order answer a new question.
2. Reasoning is not necessarily inferencing. 
3. Using architectural priors should not be discounted.
4. Attention based models work very similar to tree based models.
5. RNNs can be thought as currying functions. They take one argument and returns a function that takes one argument until the EOL token. Then it just returns a vector.

### What I still do not understand?

1. How does the CU, RU and WU work?
2. Highway gate?

### Ideas to pursue

1. Read more about attention.
