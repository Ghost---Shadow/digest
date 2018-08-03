# Natural Language Processing: Conversational models

## [Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders](https://arxiv.org/abs/1703.10960)

**Source Code:** [github](https://github.com/snakeztc/NeuralDialog-CVAE)

**Datasets:** [github](https://github.com/snakeztc/NeuralDialog-CVAE)

**Time to read (minute):** 150+ (Dropped, too difficult)

**Easy to read?** No. The math is too symbol heavy.

**Author:** (Carneige Mellon) Tiancheng Zhao, Ran Zhao and Maxine Eskenazi

**Year of Submission:** 2017

### What problem does it solve?

Uses Conditional VAE to generate diverse dialogues

### How does it solve it?

Novel framework

Dialog managers usually take a new utterance 

Latent space is conversational intent 

1. Sample a latent var `z` from prior network `p(z|c)`
2. Generate `x` through the response decoder `p(x|z,c)`

Use Bidirectional RNN with GRU cells.

Each conversation can be represented via three variables.
1. **Dialog context** - It is composed of the dialog history (last k - 1 utterances), whether it is uttered by the same speaker (0 or 1) and meta features(topic). 
2. **Response utterance** - 
3. **Latent variable** - 

#### Dataset

#### Model

### How is this paper novel?

1. Presents a new neural dialog model
2. Knowledge Guided CVAE. Easy integration with expert knowledge which results in performance improvement and model interpretability.
3. New training method for CVAE for NLP

### Key takeaways

1. Encoder decoder methods tend to generate a large number of "*I dont know*" statements.
2. The two ways to fix the *IDK* problem is to pass the dialogue history more clearly and to improve the encoder-decoder model itself. 
3. MLE objective of an encoder decoder model is unable to appx the real world goal of the conversation.
4. LSTMs tend to ignore the latent variable.

### What I still do not understand?

1. What is MLE?
2. Read more about CVAE
3. What is dyadic conversation?
4. `p(x,z|c) = p(x|z,c) p(z|c)`
5. Stochastic Gradient Variational Bayes
6. Recognition network `q(z|x,c)`
7. Variational lower bound

### Ideas to pursue

1. Encode an image to text. Use a discriminator network to classify if it is real or generated. Feed the generated text into another generator network which takes text and generates images. Have autoencode loss between the two generators.
