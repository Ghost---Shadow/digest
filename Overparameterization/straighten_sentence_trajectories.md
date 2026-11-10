# [Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language](https://arxiv.org/abs/2311.04930)

## Meta

* Journal   - NeurIPS
* Year      - 2023
* Author    - MIT, Brain and Cognitive Sciences, McGovern Institute for Brain Research (Hosseini, Fedorenko)
* Code      - Not mentioned
* Slides    - 
* One liner - Trained autoregressive transformers make the token trajectory of a sentence straighter from the early to the middle layers. Larger and better models straighten more; untrained models do not.
* Model     - GPT2 family (distilGPT2 to GPT2-XL)
* Datasets  - Universal Dependencies (subset of 8,408 sentences), OpenWebText (8,408 sentences), BookCorpus + English Wikipedia (for the models trained from random initialisation)
* Baselines - Untrained models of the same architecture

## What problem does it solve?

- An autoregressive transformer is trained to predict the next word. Which property of its internal representation makes this prediction possible?
- Hypothesis from vision (Hénaff et al., 2019): a system that must predict makes the trajectory of its input straighter in its representation space. On a straight trajectory, the next state can be found by linear extrapolation
- The paper tests this "straightening" hypothesis on language models. The tokens of a sentence are the frames of the "video"

## How does it solve it?

This is an analysis paper. It has no new training method.

### Setup

1. Take a sentence with tokens `w_1 ... w_n`
2. Send it through the model. For each layer `p`, get the activation of each token: `x_1 ... x_n`. These points make the trajectory of the sentence in layer `p`
3. Calculate the difference vector between adjacent tokens
4. Curvature at one token = the angle between two adjacent difference vectors
5. Curvature of the sentence = the average of these angles
6. Compare each layer with the first layer: a negative change means that the layer made the trajectory straighter
7. Average across all sentences

Data:

- Main set: 8,408 sentences from Universal Dependencies, 6 to 19 words each
- Check set: 8,408 sentences from OpenWebText with the same length distribution
- For the generation experiment: 5,815 sentences with 10 tokens or more, cut to 10 tokens

### Main results

1. **Curvature decreases from the early layers to the middle layers.** In trained GPT2-XL there is a large decrease. Past the middle layers the curvature goes up again on average. The untrained GPT2-XL shows no systematic change
2. **Better models straighten more.** Larger GPT-2 models have a larger decrease of curvature. Models trained on more data have a larger decrease
3. **Sentences that the model generates are straighter than the real sentences.** The model continuation and the real continuation have similar curvature in the early layers. From about layer 7 the model continuation has a larger decrease. The gap is largest near layer 20
4. **Curvature goes with surprisal.** In the middle layers of the trained model, sentences with higher surprisal have higher curvature. The correlation is largest near layer 20. The untrained model shows no relation, but layer 0 is an exception

Reference value: the curvature at the input layer is about 120 degrees. A random Gaussian trajectory in 1600 dimensions also has about 120 degrees.

### Equations

#### Difference vector

`v_k = x_(k+1) − x_k`

#### Curvature at token k (layer p)

`c_k = arccos( (v_(k+1) · v_k) / (|v_(k+1)| · |v_k|) )`

#### Curvature of a sentence and change against the first layer

```python
def curvature(X):                       # X: activations of one sentence in one layer, shape [n, d]
    V = X[1:] - X[:-1]                  # difference vectors
    V = V / norm(V, axis=1)             # only the direction is used
    cos = sum(V[1:] * V[:-1], axis=1)
    return mean(arccos(cos))            # average angle

delta_C[p] = mean([curvature(X[s][p]) - curvature(X[s][1]) for s in sentences])
```

- `0 degrees` - a straight line
- `delta_C[p] < 0` - layer `p` is straighter than the first layer
- The length of the difference vectors is not used

### Model

- **Main model:** GPT2-XL, pre-trained (48 layers, 1600 dimensions, 25 heads, 1,558M parameters)
- **Size series (pre-trained):** distilGPT2 (6 layers, 82M), GPT2 (12 layers, 117M), GPT2-Medium (24 layers, 345M), GPT2-Large (36 layers, 774M), GPT2-XL (48 layers, 1,558M)
- **Training-data series:** four GPT-2 models (12 layers, context 1024, batch size 128) trained from random initialisation on 1M, 10M, 100M and 1B tokens of BookCorpus + English Wikipedia (ratio 1:3)
- **Controls:** untrained (random) versions of the models
- **Supplement:** BERT-large-uncased, RWKV, Pythia-1.4b, OPT-1.3B, each trained and untrained

## How is this paper novel?

- Brings the straightening hypothesis from vision to language models
- One simple geometric measure (the average angle of the token trajectory) that separates trained from untrained models and better from worse models
- Shows that straightening is learned. Next-word prediction training does not ask for it directly
- Connects the geometry to a behaviour measure (surprisal) and to the generation of the model

## List of experiments

- **Experiment 1, layers:** trained GPT2-XL against untrained GPT2-XL. The curves separate clearly after about layer 10
- **Experiment 2, model size:** the average change of curvature has a logarithmic relation with the number of parameters
- **Experiment 2, training data:** the 1M model is almost the same as an untrained model. The 10M and 100M models straighten more and more. The 1B model is at about the same level as the 100M model. With more data, the layer with the largest decrease is deeper
- **Experiment 3, generation:** the first 3 tokens are the prompt, GPT2-XL generates 7 tokens (greedy). The result stays the same with beam search, with top-k sampling (k = 20) and with longer continuations (14 tokens)
- **Experiment 4, surprisal:** average 3-gram surprisal per sentence against curvature. The effect is larger for lexicalised surprisal and PCFG syntactic surprisal than for unigram frequency
- **Other architectures:** the pattern is there for RWKV, Pythia and OPT, and for BERT-large in its later layers. It is there only for the trained versions
- Exact values are only in the figures

### Ablation Studies

- **Remove attention modules:** the removal of attention in the early layers has the largest effect. It decreases the straightening the most
- **Untrained models:** no straightening
- **Second corpus (OpenWebText):** same result

### Efficiency analysis

Not mentioned

## Preliminaries

### Perceptual straightening

Hénaff et al. (2019): a natural video is a very curved trajectory in pixel space. In the human perceptual representation and in the V1 area of the macaque, the trajectory of the same video is straighter.

- A straight trajectory makes prediction easy: `x_(t+1) ≈ x_t + (x_t − x_(t−1))`
- AlexNet, which is trained for object classification, did not show straightening

### Surprisal

`surprisal(w) = −log P(w | context)`

- High surprisal = the word is hard to predict
- The paper uses the average 3-gram surprisal of the words in a sentence

### Curvature of a discrete trajectory

The angle between two adjacent steps. It uses only the direction of the steps.

- 0 degrees: the points are on a line
- About 90 degrees or more: typical for a random walk in a high-dimension space (the paper gets about 120 degrees for a random Gaussian trajectory)

## GPU hours

- The four GPT-2 models trained from random initialisation: 4 NVIDIA RTX A6000 GPUs, each model trained to its best validation loss, with a maximum of one week
- Total GPU hours: Not mentioned

## Key takeaways

- A trained language model makes the token trajectory of a sentence straighter in its middle layers
- Training is necessary. A random model keeps the trajectory as curved as a random walk
- More parameters and more training data give more straightening. The effect of data stops between 100M and 1B tokens
- The text that the model generates has a straighter trajectory than text from people
- Sentences that are hard to predict have more curved trajectories
- The result is a correlation. The paper did no causal test of "straightening causes good prediction"
- The curvature goes up again in the last layers. The authors relate this to the non-linear word space at the output

## What I still do not understand?

## Ideas to pursue

- **In common with the Squiggly Line Conjecture:** the paper studies the same object as the last intuition in the deck: the hidden-state trajectory across tokens in an autoregressive model. It gives a ready measure of the squiggle of that line (the average angle between adjacent steps)
- **Where it supports and where it contradicts:**
  - Contradiction: for this trajectory, good models are less squiggly, not more. Training makes the line straighter, a larger model makes it straighter still, and an easy sentence is straighter than a hard sentence. "More parameters = more squiggly = better" is false for the token trajectory of a trained model
  - Contradiction for random initialisation: the untrained model does not change the curvature across layers. The trajectory stays at the random-walk value (about 120 degrees). Random functions with more layers are not shown to be more squiggly or better here
  - Limit of the comparison: the paper measures the angle, not the arc length, and it measures a trajectory in the hidden space, not the graph of `f` across the input space. The conjecture is about coverage of the data space by the graph. A network can have a very bent graph across the input space and still make each sentence trajectory straight. The paper does not test the coverage loss `L(f, D)`
- **What the idea can borrow:**
  - Use the curvature measure as a second measure next to arc length. Arc length says how much space the line covers. Curvature says how predictable the line is. The paper says that prediction needs low curvature
  - A possible new form of the conjecture: a larger network has more capacity to unroll a very curved input trajectory into a straight one. The squiggle is a property of the data, and the parameters are the resource that removes it
  - Run the curvature measure and the arc length on the from-scratch models, across layers and across model sizes, at initialisation and after training

## Similar papers

- [On the Expressive Power of Deep Neural Networks](https://arxiv.org/abs/1606.05336)
- [Deep ReLU Networks Preserve Expected Length](https://arxiv.org/abs/2102.10492)
- [The Geometric Occam's Razor Implicit in Deep Learning](https://arxiv.org/abs/2111.15090)
- [A Universal Law of Robustness via Isoperimetry](https://arxiv.org/abs/2105.12806)
- [How many degrees of freedom do we need to train deep networks: a loss landscape perspective](https://arxiv.org/abs/2107.05802)
- Loss Landscapes are All You Need (ICLR 2023)
- [What's Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/abs/1911.13299)
- [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555)
- [Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
