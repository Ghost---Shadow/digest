# [Reverse Thinking Makes LLMs Stronger Reasoners](https://arxiv.org/abs/2411.19865)

## Meta

* Journal   - NAACL
* Year      - 2024
* Author    - UNC Chapel Hill, Google Cloud AI Research, Google DeepMind
* Code      - [google-research/RevThink](https://github.com/google-research/google-research/tree/master/RevThink)
* Slides    - 
* One liner - A teacher writes forward reasoning, a backward question and backward reasoning for each training question. The student learns all three with multi-task objectives.
* Model     - Teacher Gemini-1.5-Pro-001; students Mistral-7B-Instruct-v0.3, Gemma-7B-Instruct
* Datasets  - StrategyQA, CSQA, ARC-challenge, MATH, GSM8K, TabMWP, ANLI, Date Understanding (held-in); BoolQ, OpenbookQA, e-SNLI, GSM8K-Reversal (held-out)
* Baselines - Zero-shot, symbolic knowledge distillation, data augmentation methods

## What problem does it solve?

- LLMs are trained to reason only forward: question -> answer
- Humans also reason backward: from the answer to the question. This lets them check the solution
- Earlier backward-reasoning methods are mostly test-time verification, and mostly for math
- Question of the paper: can training on backward reasoning make the forward reasoning of a model better, in many domains, with no extra cost at test time?

## How does it solve it?

### Training flow

Stage 1: data augmentation with a teacher (black-box access, outputs only, no logits)

1. Start with a dataset of `(Q, A)` pairs
2. Forward reasoning: the teacher writes chain-of-thought `R_f` for `Q`. Keep the sample only if the answer in `R_f` is equal to the ground truth `A`
3. Backward question: the teacher gets `(Q, A)` and a few-shot prompt, and writes a backward question `Q_b`
4. Backward reasoning: the teacher answers `Q_b` and writes `R_b`
5. Consistency filter: the teacher gets `(Q, A, Q_b, R_b)` and says if the backward reasoning is consistent with the original question. Keep the sample only if the result is 1
6. Each sample that stays is a tuple `(Q, R_f, Q_b, R_b)`

Stage 2: train the student with three objectives together

1. `Q -> R_f` : forward reasoning (the standard distillation objective)
2. `Q -> Q_b` : write the backward question from the original question
3. `Q_b -> R_b` : answer the backward question

### Inference

1. Give the student only the original question
2. The student writes forward reasoning and the answer
3. No backward step at test time. The cost is the same as zero-shot

### Equations

#### Augmentation

```python
R_f = T(Q)                      # keep only if g(R_f) == A
Q_b = T(Q, A, I_bq)             # backward question
R_b = T(Q_b)                    # backward reasoning
c   = T(Q, A, Q_b, R_b, I_con)  # c in {0, 1}, keep only if c == 1
```

- `T` - teacher model
- `g` - function that extracts the final answer from the reasoning
- `I_bq` - prompt for backward-question generation (template + task-specific in-context examples)
- `I_con` - prompt for the consistency check

#### Loss

`L = (1 / 3n) Σ_{i=1..n} [ ℓ(S(Q_i), R_f_i) + ℓ(S(Q_i), Q_b_i) + ℓ(S(Q_b_i), R_b_i) ]`

```python
loss = 0
for Q, R_f, Q_b, R_b in data:
    loss += ce(S(Q), R_f)     # forward reasoning
    loss += ce(S(Q), Q_b)     # backward question generation
    loss += ce(S(Q_b), R_b)   # backward reasoning
loss = loss / (3 * len(data))
```

- `S` - student model
- `ℓ` / `ce` - cross-entropy between predicted tokens and target tokens
- The three terms have equal weight

### Model

- Teacher: Gemini-1.5-Pro-001
- Students: Mistral-7B-Instruct-v0.3 and Gemma-7B-Instruct
- Fine-tuning: LoRA, rank 32
- Learning rate: 5e-6 (Mistral), 2e-4 (Gemma)
- Epochs: 3 for MATH and GSM8K, 10 for the other datasets
- Decoding: greedy (temperature 0) with vLLM

## How is this paper novel?

- Backward reasoning is used as a training signal, not as a test-time check
- Works on commonsense, math, tabular, NLI and logic tasks, not only on math
- Two auxiliary objectives (`Q -> Q_b` and `Q_b -> R_b`) are added to the usual distillation objective
- The teacher filters the data two times: correct forward answer, and backward reasoning that is consistent with the original question
- Test-time cost stays the same as zero-shot

## List of experiments

- **Main result (average accuracy on 8 held-in datasets, Mistral / Gemma)**
  - Teacher zero-shot (Gemini-1.5-Pro): 79.76
  - Zero-shot: 50.55 / 46.80
  - SKD: 56.08 / 54.65
  - Distill Step-by-Step: 56.26 / 54.73
  - Rephrase Question: 55.41 / 53.18
  - Question Aug: 56.56 / 54.29
  - Answer Aug: 57.49 / 56.65
  - RevThink: 63.23 / 61.17
  - Abstract: 13.53% average gain over zero-shot, 6.84% over the strongest distillation baseline
- **Held-out generalization (BoolQ / OpenbookQA / e-SNLI / GSM8K-Reversal)**
  - Zero-shot: 53.18 / 70.20 / 52.27 / 17.37
  - SKD: 60.82 / 75.40 / 56.62 / 27.54
  - AnsAug: 61.74 / 76.40 / 55.54 / 28.96
  - RevThink: 63.85 / 79.60 / 61.97 / 32.05
- **Complementary with other augmentation (StrategyQA / GSM8K)**
  - RevThink: 70.97 / 60.88
  - RevThink + AnsAug: 73.56 / 61.58
- **Sample efficiency:** RevThink is better than SKD at 10%, 25%, 50% and 100% of the training data. On StrategyQA, RevThink with 10% of the data is better than SKD with 100%
- **Student size (StrategyQA):** gain at each model size. Mistral-7B + RevThink is 8.36% better than zero-shot Mistral-8x22B
- **MATH breakdown:** largest gains in prealgebra, precalculus, and counting & probability. No significant gain in number theory

### Ablation Studies

- **Objectives:** all three objectives together are best. `Q_b -> R_b` alone is the worst and can be below zero-shot. Adding `Q -> Q_b` to `Q -> R_f` helps. (Results are in a figure, no numbers in the text)
- **How to combine the objectives (StrategyQA / GSM8K / Date):**
  - Multi-task with instructions: 71.19 / 59.81 / 64.36
  - Multi-task with task prefix: 69.11 / 58.74 / 66.12
  - Joint objectives (RevThink): 73.36 / 60.62 / 70.40
- **Remove the teacher verification:** StrategyQA 70.97 -> 68.63, GSM8K 60.88 -> 59.75, ANLI 48.58 -> 47.63

### Efficiency analysis

- Training tokens: RevThink makes slightly more training tokens than Answer Augmentation, but has higher accuracy
- Test-time tokens: about the same for all methods, because the student only does forward reasoning
- The filters remove much data. Examples: MATH 7,500 -> 2,511 samples, GSM8K 7,379 -> 4,293, CSQA 9,741 -> 6,478

## Preliminaries

### Symbolic knowledge distillation (SKD)

- A large teacher LLM writes chain-of-thought reasoning for each training question
- A small student is fine-tuned to write the same reasoning from the question
- Only text goes from teacher to student. No logits are necessary, so the teacher can be a black-box API

### Backward reasoning / backward question

- Forward: "Emma has 2 apples, Jack has 3. How many in total?" -> 5
- Backward: "Emma and Jack have 5 apples in total. Emma has 2. How many does Jack have?" -> 3
- The backward question uses the answer of the forward question as an input and asks for one of the original inputs
- If the forward answer and the backward answer agree, the reasoning is probably correct

### LoRA

- Freeze the weights `W` of the model
- Learn a low-rank update: `W' = W + B @ A`, where `A` and `B` have rank `r` (here `r = 32`)
- Only `A` and `B` are trained, so fine-tuning is cheap

## GPU hours

Not mentioned

## Key takeaways

- Training on the inverse task makes the forward task better. The model does not have to do the inverse task at test time
- The backward data must be related to the forward data. Backward reasoning alone (`Q_b -> R_b`) is worse than zero-shot in some cases
- Filtering by a consistency check gives a small but steady gain (1-2 points)
- The method is data-efficient: 10% of the data can be better than the baseline with 100% (StrategyQA)
- The method needs a strong teacher and labeled `(Q, A)` data. The student does not make its own backward questions

## What I still do not understand?

## Ideas to pursue

- **In common with the causal-consistency objective:** both train the model on a pair of inverse tasks (RevThink: forward question / backward question; the idea: "Why?" / "Therefore?") with plain next-token prediction, and both get a gain in the forward direction from the inverse direction. The RevThink consistency check `c` is a cycle check: does the backward pair agree with the forward pair?
- **What RevThink lacks:** it is supervised distillation. It needs labeled `(Q, A)` pairs and a stronger teacher (Gemini) to write and to filter the backward data. The cycle is not closed by the student itself: the student never uses its own output as the input of the inverse step. It operates on QA tasks, not on raw pre-training lines, so it does not enforce consistency across documents. It has no distractor / multiple-choice loss
- **What the idea can borrow:**
  - The three-term joint loss. Add a term that writes the inverse prompt (`line -> "Why?" answer`) together with the forward term (`answer -> "Therefore?" -> line`), with equal weights. RevThink shows the joint form is better than multi-task with prefixes
  - The ablation result that the inverse task alone can hurt. Keep a normal next-token loss on the original line in the mix
  - The filter before the loss. Discard an abduced answer if the "Therefore?" step cannot recover the original line
  - GSM8K-Reversal as a held-out test of inverse reasoning

## Similar papers

- [Deductive Closure Training](https://arxiv.org/abs/2401.08574)
- [Reasoning to Learn from Latent Thoughts](https://arxiv.org/abs/2503.18866)
- [Cycle-Instruct](https://arxiv.org/abs/2508.16100)
- [Time-Reversal Provides Unsupervised Feedback to LLMs](https://arxiv.org/abs/2412.02626)
- [Flip-Flop Consistency](https://arxiv.org/abs/2510.14242)
- [LEDOM](https://aclanthology.org/2026.acl-long.975)
- [The Reversal Curse](https://arxiv.org/abs/2309.12288)
- [Maieutic Prompting](https://arxiv.org/abs/2205.11822)
