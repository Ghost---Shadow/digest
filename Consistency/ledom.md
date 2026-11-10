# [LEDOM: Reverse Language Model](https://aclanthology.org/2026.acl-long.975)

## Meta

* Journal   - ACL
* Year      - 2025 (arXiv), published 2026
* Author    - Xunjian Yin, Sitao Cheng, Yuxi Xie, Xinyu Hu, Li Lin, Xinyi Wang, Liangming Pan, William Yang Wang, Xiaojun Wan (Peking University, Princeton, University of Waterloo, UC Santa Barbara, National University of Singapore)
* Code      - [github.com/Arvid-pku/LEDOM](https://github.com/Arvid-pku/LEDOM)
* Slides    - 
* One liner - Train a language model purely right-to-left, then use it to score the outputs of forward models (Reverse Reward).
* Model     - LEDOM 2B and 7B (trained from scratch)
* Datasets  - Pre-training: DCLM-Baseline (general), MAP-Neo math data, MAP-Neo code data. Fine-tuning: OpenMathInstruct-2. Evaluation: GSM8K, HellaSwag, HumanEval, NQ-Open, OpenBookQA, BoolQ, TriviaQA, WinoGrande, MMLU, MATH-500, AIME 2024, AMC 2023
* Baselines - Forward models: DeepSeekMath-7B, OpenMath2-8B, QwenMath-7B

arXiv: [2507.01335](https://arxiv.org/abs/2507.01335)

## What problem does it solve?

- All large LMs are trained left-to-right. The right-to-left factorization is equally valid by the chain rule, but nobody trained it at scale.
- Question 1: what abilities does a model get when it predicts the past from the future?
- Question 2: can a forward model and a reverse model be used together?
- Application: a forward model can generate a hallucinated reasoning chain that has a high forward likelihood. The reverse model does a check if the output can reconstruct the input prompt.

## How does it solve it?

### Training flow

1. Collect a corpus of 435B tokens (general text, math, code).
2. Tokenize each text (SentencePiece).
3. Reverse the token order: `(x_1, ..., x_T)` becomes `(x_T, ..., x_1)`.
4. Train a standard causal Transformer decoder on the reversed sequences with next-token prediction. Thus the model predicts the token before, given all the tokens after.
5. Train a forward model (FLM) with the same architecture, tokenizer and data for comparison.
6. For the math application: fine-tune LEDOM on 100,000 examples of OpenMathInstruct-2 with the same reverse objective. This model is the reward model.

### Inference

LEDOM alone:

1. Reverse the tokens of the input (for example a conclusion or an answer).
2. Let LEDOM generate. The output is the text that comes before the input, in reversed order.
3. Reverse the output tokens to read it.
4. For few-shot benchmarks: reverse the question, the steps and the answer of each demonstration. The test question goes at the start of the reversed prompt. The marker strings (`Question`, `Step`, `Answer`) are not reversed.

Reverse Reward, response level (Best-of-N):

1. The forward model generates `N` candidate responses `y` for the prompt `x`.
2. LEDOM reads the reversed concatenation `[reverse(y); reverse(x)]` and gives `P_RLM(x | y)`.
3. Calculate the bidirectional score for each candidate.
4. Select the candidate with the highest score.

Reverse Reward, step level (beam search):

1. Keep `k` active beams. Each beam is a partial chain of reasoning steps.
2. For each beam, the forward model generates `n` candidate next steps. This gives `n * k` candidates.
3. Score each candidate with the bidirectional score.
4. Keep the top `k` candidates as the new beams.
5. A step ends at a double newline (`\n\n`). A chain ends when it contains `\boxed{}`.
6. Return the completed chain with the highest final score.

### Equations

#### Forward and reverse factorization

```python
p_flm(x) = prod([p(x[t] | x[:t])   for t in range(T)])   # left-to-right
p_rlm(x) = prod([p(x[t] | x[t+1:]) for t in range(T)])   # right-to-left
```

- The two decompose the same joint `P(x)`.
- The two models have the same theoretical optimum but learn different internal representations.

#### Loss

`L_RLM = -E_x[ sum_t log p(x_t | x_{t+1:T}) ]`

#### Bayes relation between the two directions

`P(x | y) = P(y | x) * P(x) / P(y)`

- `x` - prompt, `y` - response
- FLM estimates `P(y | x)`. RLM estimates `P(x | y)`.

#### Directional entropy asymmetry

`H(Y | X) - H(X | Y) = H(Y) - H(X)`

- If responses are longer and have more variance than prompts, then `H(X | Y) < H(Y | X)`.
- The paper says: thus the reverse score is a stronger signal to rank candidates.

#### Reverse Reward

```python
# how well does the response y reconstruct the prompt x
reverse_reward = prod([p_rlm(x[t] | x[t+1:], y) for t in range(T)])   # = p_rlm(x | y)

# bidirectional score, lam in [0, 1]
score = p_flm(y | x) ** (1 - lam) * reverse_reward ** lam
```

With Bayes and logarithms:

```python
log_score = log p(y | x) - lam * log p(y) + c            # c = lam * log p(x), constant
log_score = (1 - lam) * log p(y | x) + lam * pmi(x, y) + c
pmi(x, y) = log p(y | x) - log p(y)
```

- The term `- lam * log p(y)` penalizes responses that are generic and do not depend on the prompt.
- This is the opposite sign from the classic noisy-channel rule `argmax_y P(x | y) P(y)`, which rewards fluent `y`.

#### Proposition 1 (posterior margin dominates forward noise)

```python
delta_f = log p_flm(y_good | x) - log p_flm(y_bad | x)
delta_r = log p_rlm(x | y_good) - log p_rlm(x | y_bad)

# y_good is ranked above y_bad if and only if
(1 - lam) * delta_f + lam * delta_r > 0

# if delta_r > delta_f, this is true for every
lam > max(0, -delta_f / (delta_r - delta_f))
```

- Hypothesis ("posterior degradation"): a hallucinated chain adds steps that are not in the premises. Thus it is harder to reconstruct the prompt from it, and `p_rlm(x | y)` is lower.

### Model

| Model | Layers | Heads | d_model | FFN dim | KV heads |
|---|---|---|---|---|---|
| 2B | 18 | 8 | 2048 | 16384 | 1 |
| 7B | 28 | 16 | 3072 | 24576 | 16 |

- Transformer decoder, Multi-Query / Grouped-Query Attention, RoPE, RMSNorm, SwiGLU
- Context window: 8192 tokens
- Embeddings and output weights untied, no bias in linear layers, no dropout
- Forward baselines for Reverse Reward: DeepSeekMath-7B, OpenMath2-8B, QwenMath-7B

## How is this paper novel?

- The paper claims to be the first open-source, purely reverse-trained autoregressive LM at scale (2B and 7B parameters, 435B tokens), with matched forward models.
- Earlier work on reverse generation: regularization of forward models (Serdyuk et al. 2017, Zhang et al. 2019), small diagnostic reverse models (Pfau et al. 2023), two-stage forward-then-reverse training (Golovneva et al. 2024).
- Nearest work: TRLM (Varun et al. 2025) used a small reverse model for unsupervised feedback and Best-of-N reranking. LEDOM adds a Bayesian basis (noisy channel, Proposition 1) and step-level verification.

## List of experiments

Case studies (qualitative, one example each):

- Abductive reasoning: input "That is why Mike gave up his job and started his own business." The model writes the backstory before it.
- Story generation from the end of the story.
- Math: input "+2 = 6." The model writes the expression before it.
- Question generation: from an answer with its reasoning, the model writes the question.
- Reversal curse: input "is the author of Harry Potter." The model writes "J.K. Rowling". The paper says this is an illustration, not a full evaluation on the benchmarks of Berglund et al.
- Safety: LEDOM generated harmful content from a prompt that would trigger the safeguards of a forward model.

Benchmark comparison with the forward model (accuracy % or Pass@1):

| Model | GSM8K | HellaSwag | HumanEval | NQ-Open | OpenBookQA | BoolQ | TriviaQA | WinoGrande | MMLU |
|---|---|---|---|---|---|---|---|---|---|
| FLM-2B | 2.96 | 57.37 | 8.54 | 11.55 | 23.00 | 59.69 | 40.22 | 55.17 | 24.40 |
| LEDOM-2B | 1.74 | 54.45 | 2.44 | 5.57 | 24.80 | 61.35 | 19.82 | 53.28 | 23.55 |
| FLM-7B | 16.83 | 67.06 | 13.41 | 19.25 | 25.20 | 65.69 | 57.28 | 58.33 | 29.26 |
| LEDOM-7B | 1.74 | 62.41 | 1.22 | 14.27 | 22.40 | 37.77 | 39.06 | 53.04 | 24.95 |

- LEDOM is competitive on semantic tasks (BoolQ and OpenBookQA at 2B).
- LEDOM is weak on forward-causal tasks: code generation (HumanEval) and factual recall (NQ-Open, TriviaQA).
- The gap becomes larger at 7B (BoolQ 37.77 against 65.69).

Reverse Reward on math (accuracy %):

| Model | Strategy | MATH-500 | GSM8K | AIME 2024 | AMC 2023 |
|---|---|---|---|---|---|
| DeepSeekMath | Greedy | 42.0 | 81.8 | 10.0 | 12.5 |
| DeepSeekMath | Best-of-N (random) | 40.7 | 81.1 | 8.97 | 18.6 |
| DeepSeekMath | Best-of-N (Reverse Reward) | 43.6 | 84.1 | 13.3 | 27.5 |
| QwenMath | Greedy | 78.0 | 95.6 | 16.7 | 55.0 |
| QwenMath | Best-of-N (random) | 73.9 | 94.7 | 11.3 | 48.3 |
| QwenMath | Best-of-N (Reverse Reward) | 80.8 | 96.1 | 23.3 | 57.5 |
| OpenMath2 | Greedy | 64.0 | 89.8 | 10.0 | 40.0 |
| OpenMath2 | Best-of-N (random) | 56.2 | 87.1 | 10.0 | 24.8 |
| OpenMath2 | Best-of-N (Reverse Reward) | 65.0 | 91.0 | 16.7 | 40.0 |
| OpenMath2 | Beam search (Reverse Reward) | 65.4 | 91.8 | 6.7 | 42.5 |

- The largest gains are on the competition-level problems (AIME, AMC).
- Beam search is worse than greedy on AIME 2024 for OpenMath2 (6.7 against 10.0). Step-level pruning on long chains can discard beams that are partially correct.
- Case study: on a GSM8K problem, the top forward candidate ignores the constraint "restart the download from the beginning". Reverse Reward selects the candidate that uses the constraint.

### Ablation Studies

- Sampling size `N` from 1 to 64 in step-level beam search (MATH-500 and GSM8K): the paper says the accuracy improves monotonically with `N` (numbers only in a figure).
- Granularity: response-level reranking against step-level beam search (only with OpenMath2, because of the computational cost).
- Best-of-N random is the control that isolates the reranking effect from the sampling diversity.
- Inference parameters (appendix D.2.3): Best-of-N with `N = 4`, beam width `k = 4`, `n = 3` new steps for each beam. But the caption of the main results table says Best-of-N samples 64 candidates. The paper does not explain the difference.
- The value of `lam` and an ablation of `lam` are not given.
- No comparison with learned verifiers (outcome or process reward models). The authors list this as a limitation.

### Efficiency analysis

- Reverse Reward adds one RLM forward pass for each candidate.
- Best-of-N with `N = 64`: 64 reverse evaluations for each problem. Each is comparable in cost to a forward generation pass, because the RLM has the same architecture.
- Step-level beam search scores only the `k` active beams at each step, not all `N` candidates.
- Training dynamics: the RLM converges more slowly and has a higher final training loss than the FLM. Reasons given: (1) the hardest predictions are at the end of the reversed sequence, where the model must recover topic sentences and named-entity introductions from generic later text, (2) position encodings and initialization were tuned for left-to-right training, (3) finite capacity.

## Preliminaries

### Chain rule in two directions

A joint probability can be factorized in any order.

```python
p(a, b, c) = p(a) * p(b | a) * p(c | a, b)     # left-to-right
p(a, b, c) = p(c) * p(b | c) * p(a | b, c)     # right-to-left
```

- The two are exact and equal.
- A model with finite capacity learns different approximations for each order.

### Noisy channel model

- Classic rule (Shannon): to find the source `y` from an observation `x`, use `argmax_y P(x | y) * P(y)`.
- `P(x | y)` is the channel model (how well `y` explains `x`). `P(y)` is the prior (how fluent `y` is).
- In this paper the reverse model is the channel model. The sign of the prior is opposite: generic responses with high `P(y)` are penalized.

### Pointwise mutual information (PMI)

`pmi(x, y) = log p(y | x) - log p(y) = log p(x | y) - log p(x)`

- High PMI: `y` is much more likely with `x` than without `x`. Thus `y` is specific to the prompt.
- Low PMI: `y` is a generic response that fits any prompt.

## GPU hours

- Hardware: 8 Oracle Cloud nodes, each with 8 NVIDIA A100 80GB (64 GPUs total).
- Each 7B model: approximately 628 hours, approximately 51,900 iterations.
- Each 2B model: approximately 307 hours.
- The paper gives wall-clock hours. If all 64 GPUs were used for the full time (my calculation, not a number in the paper): 628 * 64 = 40,192 GPU hours for a 7B model and 307 * 64 = 19,648 GPU hours for a 2B model.
- Training settings: AdamW (beta1 0.9, beta2 0.95), peak learning rate 2e-4 with cosine decay to 2e-5, 2000 warmup iterations, weight decay 0.1, gradient clip 1.0, global batch 1024 sequences of 8192 tokens, BF16.
- Math fine-tuning of LEDOM: 4 GPUs, 2 epochs, 100,000 examples, max length 1024, learning rate 1e-5, AdamW, no weight decay. Hours not mentioned.

## Key takeaways

- A reverse LM does abduction as its default mode: given a conclusion, it generates the premises.
- The reverse direction is harder to learn on English text (slower convergence, higher loss).
- Forward and reverse models fail on different tasks and make different errors. This is the reason to combine them.
- `P(prompt | response)` from the reverse model is an unsupervised verifier. It needs no labelled data.
- Hallucinated chains are harder to invert back to the question.
- Step-level verification is not always better. On long chains it can prune correct beams too early.
- Safety training for forward models does not transfer to the reverse direction.

## What I still do not understand?

## Ideas to pursue

- **In common:** LEDOM is "Why?" in model form. Its input is a conclusion and its output is the text that leads to it (the abductive reasoning case study). Reverse Reward is a cycle check: the forward model goes from `x` to `y`, and the reverse model does a check that `y` leads back to `x`. This is the same structure as "Why?" then "Therefore?" with the original line as the target. The score `P_RLM(x | y)` is the same quantity as the likelihood of the original line given the abduced cause.
- **What it lacks:** The cycle is only used at inference to rerank. It is not a training loss, and no gradient goes from the reverse check to the forward model. It needs two models with different weights. The causal-consistency objective uses one model for the two directions. There is no distractor: candidates are ranked against each other, not against a negative line. All pairs are in one sequence (one document). There is no cross-document consistency and no repeated abduction that makes a graph to a root cause.
- **What to borrow:** (1) The PMI term: penalize "why" answers with high marginal `P(y)`. A generic "why" answer (one that fits any line) must not get credit. This is a possible weight or filter for the cycle loss. (2) The posterior degradation hypothesis as a test: a hallucinated "why" answer will reconstruct the original line worse than a correct one. Proposition 1 gives the condition for when the reverse signal is stronger than the forward signal. (3) A reverse model (LEDOM is open-source) can generate the "why" answers or the hard distractor lines, as an alternative to the same forward model. (4) The warning from the AIME result: step-level pruning can remove correct chains early. Thus do not filter the "why" graph too strongly at each hop.

## Similar papers

- [Deductive Closure Training](https://arxiv.org/abs/2401.08574)
- [RevThink](https://arxiv.org/abs/2411.19865)
- [Reasoning to Learn from Latent Thoughts](https://arxiv.org/abs/2503.18866)
- [Cycle-Instruct](https://arxiv.org/abs/2508.16100)
- [Time-Reversal Provides Unsupervised Feedback to LLMs](https://arxiv.org/abs/2412.02626)
- [Flip-Flop Consistency](https://arxiv.org/abs/2510.14242)
- [The Reversal Curse](https://arxiv.org/abs/2309.12288)
- [Maieutic Prompting](https://arxiv.org/abs/2205.11822)
