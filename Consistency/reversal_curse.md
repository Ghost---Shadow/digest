# [The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"](https://arxiv.org/abs/2309.12288)

## Meta

* Journal   - ICLR
* Year      - 2023 (arXiv, September 2023), published 2024
* Author    - Lukas Berglund (Vanderbilt), Meg Tong, Max Kaufmann (UK AI Safety Institute), Mikita Balesni (Apollo Research), Asa Cooper Stickland (NYU), Tomasz Korbak (Sussex), Owain Evans (Oxford)
* Code      - [github.com/lukasberglund/reversal_curse](https://github.com/lukasberglund/reversal_curse)
* Slides    - 
* One liner - An LM fine-tuned on "A is B" does not learn "B is A". The likelihood of the correct name is no higher than that of a random name.
* Model     - GPT-3 (350M to 175B), Llama-1 (7b, 13b, 30b), GPT-3.5, GPT-4
* Datasets  - Synthetic fictitious-celebrity facts (made with GPT-4), 1573 real celebrity child-parent pairs (top 1000 IMDb celebrities), synthetic question-answer instructions
* Baselines - None (analysis paper)

## What problem does it solve?

- It does not solve a problem. It is an analysis paper that shows a failure of auto-regressive LLMs.
- A model that learns "A is B" from training data does not learn "B is A".
- Example: the model learns "Valentina Tereshkova was the first woman to travel to space". It cannot answer "Who was the first woman to travel to space?" better than chance.
- The failure occurs only for facts in the weights. If "A is B" is in the context window, the model can infer "B is A".

## How does it solve it?

There is no new training method. The paper has three experiments that show the failure.

### Experiment setup

#### Experiment 1: reverse descriptions of fictitious celebrities

1. Use GPT-4 to make pairs of `<name>` and `<description>` for persons who do not exist. Example: "Daphne Barrington is the director of 'A Journey Through Time'".
2. Make 30 facts. Paraphrase each fact 30 times. This gives 900 documents for each subset.
3. Make three subsets:
   - NameToDescription: the name comes before the description.
   - DescriptionToName: the description comes before the name.
   - Both: the fact is given in the two orders, in different documents. This subset is only auxiliary training data.
4. Fine-tune the model on all subsets (3600 training examples).
5. Test with held-out prompts (10 held-out paraphrases for each fact and each direction).
6. Test in the same direction as the training and in the reverse direction.
7. Measure exact-match accuracy at temperature 0.
8. For NameToDescription, also compare the log-probability of the correct name with the log-probability of a random name.

#### Experiment 2: real celebrities and their parents

1. Take the top 1000 celebrities from IMDb.
2. Ask GPT-4 for the parents of each celebrity. This gives 1573 child-parent pairs.
3. For each pair, ask the parent question: "Who is [name]'s [mother/father]?"
4. For each pair, ask the child question: "Name a child of [name]."
5. Chat models (GPT-4, gpt-3.5-turbo): ask 10 times at temperature 1. It is a success if the model is correct a minimum of one time.
6. Llama-1 base models: measure the likelihood of the correct completion.
7. There is no fine-tuning in this experiment.

#### Experiment 3: reverse instructions

1. Make question-answer pairs. Example: "What was your favorite book as a child?" -> "Charlotte's Web".
2. Make two datasets of instructions, 1100 pairs each:
   - QuestionToAnswer: "Answer `<question>` with `<answer>`."
   - AnswerToQuestion: "Answer with `<answer>` when you see `<question>`."
3. Fine-tune Llama-1 on the instructions.
4. Test with the prompt "Q: `<question>` A:" at temperature 0.

### Training flow

- Standard fine-tuning only (next-token prediction). No new loss.
- GPT-3 fine-tuning through the OpenAI API: 10 epochs, sweep of learning rate multiplier (0.05, 0.1, 0.2, 0.4) and batch size (1, 2, 4, 8, 16). The loss is not masked on the prompts.
- Llama-1 7b full fine-tuning: sweep of batch size (1, 4, 16) and learning rate (1e-6, 2e-6, 1e-5, 2e-5), Adam.
- Experiment 3: Llama-1 sweep with 5 epochs, then final runs of 20 epochs with 5 seeds for each model size.

### Inference

- Greedy generation (temperature 0) and exact match for Experiments 1 and 3.
- 10 samples at temperature 1 for the chat models in Experiment 2.

### Equations

Definition of the Reversal Curse. A model is trained on "`<name>` is `<description>`". Let `n` be the name and `d` the description.

```python
# The Reversal Curse holds if the correct name is not more likely
# than a random name n_r when the model sees the description
p_llm(n | d) <= p_llm(n_r | d)
```

- The paper tests this with a paired t-test and a Kolmogorov-Smirnov test on the log-probabilities.
- Result: no detectable difference between the correct name and a random name.

### Model

- GPT-3 base models: 350M, 1.3B, 6.7B, 175B parameters (Experiment 1)
- Llama-1 7b (Experiment 1), Llama-1 7b, 13b, 30b (Experiments 2 and 3)
- GPT-4 and gpt-3.5-turbo (Experiment 2, no fine-tuning)

## How is this paper novel?

- It shows the failure with a controlled fine-tuning experiment on fictitious facts. The model cannot know these facts from pre-training.
- It shows that the reverse accuracy is not only low. The log-probability of the correct name is not higher than that of a random name.
- It shows evidence of the same failure in production models (GPT-4) on real facts.
- Related work that the paper refers to:
  - Grosse et al. (2023) used influence functions. Training examples in the same order have much more influence than examples in the reverse order.
  - Allen-Zhu and Li (2023) found the same failure in models trained from scratch on synthetic data (concurrent work).
  - Meng et al. (2023): model edits are not bidirectional.

## List of experiments

- Experiment 1 (GPT-3 175B, exact match, mean ± SD of 3 seeds):
  - NameToDescription: same direction 50.0 ± 2.1, reverse direction 0.0 ± 0.0
  - DescriptionToName: same direction 96.7 ± 1.2, reverse direction 0.1 ± 0.1
- Experiment 1 log-probability: no statistically significant difference between the correct name and a random name in all 12 runs (4 model sizes, 3 seeds).
- Experiment 2:
  - GPT-4 gives the parent for 79% of the celebrities.
  - GPT-4 gives the child for 33% of the pairs (text and abstract). The note of the figure gives 28%. The paper does not explain the difference.
  - Llama-1 models are also much better at the parent than at the child (numbers only in a figure).
- Experiment 3:
  - QuestionToAnswer (same order as the test): above 80% for all Llama-1 models.
  - AnswerToQuestion (reverse order): below 7%.

### Ablation Studies

- Hyperparameter sweep (GPT-3 350M and Llama-1 7b): the reverse accuracy stays near 0% for all settings.
- Model size (350M to 175B): no improvement.
- Auxiliary "Both" subset (facts in the two orders): does not help the model to reverse the other facts.
- Paraphrases (30 for each fact): do not help.
- In-context test (3-shot, temperature 0): 96.67% to 100%. The curse does not occur when the fact is in the context.
- Larger dataset (100 facts for each subset, 100 paraphrases, 40,000 documents, GPT-3 350M): reverse accuracy is 0.0.
- Prompt tuning on Llama-1 7b: the reverse test set and a shuffled reverse test set get equal scores. Thus there is no reverse association.

### Efficiency analysis

Not mentioned

## Preliminaries

### Auto-regressive (left-to-right) language model

The model learns the probability of each token from the tokens before it.

```python
p(x) = prod([p(x[t] | x[:t]) for t in range(T)])
```

- The loss for "A is B" only trains `p(B | A)`.
- No term in the loss trains `p(A | B)`.
- The paper says the gradient update is "myopic": it depends on the logits of B given A. It does not depend on a future need to predict A from B.

### Symmetric relation

- The relation "is" (identity) is symmetric: if "A is B" is true then "B is A" is true.
- A human who knows one direction can deduce the other direction.
- The paper says the curse is not strictly a failure of logical deduction, because the model can do the deduction in context. It is a failure to generalize beyond the training data (a failure of meta-learning).

### Exact match and log-probability tests

- Exact match: generate at temperature 0 and compare the output with the correct answer.
- Exact match can be 0% while the model still has a weak preference for the correct answer.
- Thus the paper also compares `log p(correct name | description)` with `log p(random name | description)`. If the two are equal, the model learned nothing in the reverse direction.

## GPU hours

- Not given as a total.
- OpenAI API cost: approximately $100 each for Experiments 1 and 2.
- Llama training: Nvidia A100 GPUs on the Center for AI Safety cluster.
- Llama-1 30b: eight A100 GPUs, 20 to 160 minutes for each epoch (depends on batch size).

## Key takeaways

- Next-token prediction stores a fact in one direction only.
- More data of the same direction, more paraphrases, larger models and other hyperparameters do not repair this.
- A fact in the context window does not have this problem. The problem is in what the weights store.
- Facts that occur in the two orders in the corpus are safe. Rare (long-tail) facts that occur in one order are at risk.
- Knowledge in an LLM is not a consistent set of facts. It is a set of directed associations.

## What I still do not understand?

## Ideas to pursue

- **In common:** The paper shows that a forward-only loss does not give the inverse direction for free. "Why" (effect -> cause) and "Therefore" (cause -> effect) are an inverse pair in the same way as "A is B" and "B is A". Thus the paper is direct evidence that the inverse direction must be trained explicitly. The causal-consistency objective does this: the "Therefore?" step trains the direction opposite to the "Why?" step.
- **What it lacks:** The paper only measures the failure. It has no loss and no repair. It tests only the identity relation between a name and a description, not causal relations between lines of different documents. The "Both" subset shows that some examples in the two orders do not transfer to other facts. Thus a cycle loss on sampled lines will possibly repair only the lines it samples.
- **What to borrow:** The test design. (1) Use fictitious facts so that pre-training cannot leak the answer. (2) Compare the log-probability of the correct option with a random option, not only exact match. This is almost the same as the "original line against a distractor line" step of the idea. (3) Report the same direction and the reverse direction separately. (4) Test if the cycle loss on some lines improves consistency on held-out lines, because the "Both" result says this transfer is not automatic.

## Similar papers

- [Deductive Closure Training](https://arxiv.org/abs/2401.08574)
- [RevThink](https://arxiv.org/abs/2411.19865)
- [Reasoning to Learn from Latent Thoughts](https://arxiv.org/abs/2503.18866)
- [Cycle-Instruct](https://arxiv.org/abs/2508.16100)
- [Time-Reversal Provides Unsupervised Feedback to LLMs](https://arxiv.org/abs/2412.02626)
- [Flip-Flop Consistency](https://arxiv.org/abs/2510.14242)
- [LEDOM: Reverse Language Model](https://aclanthology.org/2026.acl-long.975)
- [Maieutic Prompting](https://arxiv.org/abs/2205.11822)
