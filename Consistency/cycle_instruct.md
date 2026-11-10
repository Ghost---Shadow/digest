# [Cycle-Instruct: Fully Seed-Free Instruction Tuning via Dual Self-Training and Cycle Consistency](https://arxiv.org/abs/2508.16100)

## Meta

* Journal   - EMNLP
* Year      - 2025
* Author    - Zhejiang University, University of Science and Technology of China, Tianjin University (Zhanming Shen, Hao Chen, Yulei Tang, Shaolin Zhu, Wentao Ye, Xiaomeng Hu, Haobo Wang, Gang Chen, Junbo Zhao)
* Code      - Not mentioned
* Slides    - 
* One liner - Two models, question-to-answer and answer-to-question, label raw unpaired text for each other. The cycle-consistency signal replaces the labeled seed set of instruction back-translation.
* Model     - Llama-3.1-8B (LoRA)
* Datasets  - Alpaca-GPT4, Dolly-15k, Medical-Alpaca, OASST-1, WikiHow (training data). MMLU, BBH, CRASS, DROP, AlpacaEval (evaluation).
* Baselines - Vanilla, All-SFT, SFT-80, seed-based back-translation (Rand-k%, Clust-k%)

## What problem does it solve?

- Instruction tuning usually needs human-written (instruction, answer) pairs or a strong teacher model
- Instruction back-translation decreases this cost, but it still needs a seed set of labeled pairs
- The seed set costs effort, transfers its bias to the generated data, and leaves most unlabeled text unused
- The paper makes instruction-tuning data from raw text with 0% labeled seed data

## How does it solve it?

### Training flow

1. Split raw documents into paragraphs at blank lines
2. A paragraph that contains a `?` goes into the question pool. All other paragraphs go into the answer pool. The two pools are not paired.
3. A fixed "prompter" prompt rewrites each question paragraph into one self-contained question `q_i`
4. A fixed "assistant" prompt cleans each answer paragraph into a coherent answer `a_j`, with no new information
5. Start two models from the same base model: the forward model `M_QA` (question -> answer) and the backward model `M_AQ` (answer -> question)
6. `M_QA` generates a pseudo-answer `a_hat_i` for each real question `q_i`
7. Train `M_AQ` on the pairs `(a_hat_i -> q_i)`. The target is always real text.
8. `M_AQ` generates a pseudo-question `q_hat_j` for each real answer `a_j`
9. Train `M_QA` on the pairs `(q_hat_j -> a_j)`. The target is always real text.
10. Repeat steps 6-9 for a fixed number of cycles (optional). The main tables use one round.
11. The final dataset is the union of `(q_i, a_hat_i)` and `(q_hat_j, a_j)`
12. Optional filter (Cycle-Filt): remove the 5% of pairs with the worst reconstruction (see Equations)
13. Fine-tune the base model on the final dataset

### Inference

1. The model fine-tuned on the final dataset answers instructions as a usual instruction-tuned model
2. There is no cycle step at inference

### Equations

#### Baseline: instruction back-translation (needs a seed set S)

`L_inv = - sum over (q,a) in S of log p(q | a)`

`L_BT = E over (q_hat, a) of [ - log p(a | q_hat) ]`

- First train a backward model on the seed pairs, then use it to label unlabeled answers

#### Cycle-consistency loss (the concept)

`L_cycle = E_q[ l(q, F_AQ(G_QA(q))) ] + E_a[ l(a, G_QA(F_AQ(a))) ]`

- `G_QA` - forward model, `F_AQ` - backward model
- `l` - usually the negative log-likelihood of the original sequence
- If you go question -> answer -> question, you must get the original question again. The same for answers.

#### The two losses that are trained

```python
# step 7: backward model learns to rebuild the real question from the pseudo-answer
L_AQ = -mean([log_p(q[i], given=a_hat[i], model=M_AQ) for i in range(N_Q)])

# step 9: forward model learns to rebuild the real answer from the pseudo-question
L_QA = -mean([log_p(a[j], given=q_hat[j], model=M_QA) for j in range(N_A)])
```

- The paper gives no weight between the two losses. Each model is trained separately.

#### Cycle-consistency filter (Cycle-Filt)

```python
# 1. reconstruct the real side from the pseudo side with the opposite model
q_rec[i] = M_AQ(a_hat[i])
a_rec[j] = M_QA(q_hat[j])

# 2. distance between original and reconstruction in embedding space
d[i] = l2(phi(q[i]), phi(q_rec[i]))
d[j] = l2(phi(a[j]), phi(a_rec[j]))

# 3. k-means on the real sides, k = 200
# 4. in each cluster, drop the 5% of points with the largest distance
```

- `phi` - sentence encoder (the LLaMA3 encoder, not fine-tuned)
- The result keeps approximately 95% of the data

### Model

- Llama-3.1-8B base
- LoRA: rank 8, alpha 16, dropout 0.05
- Learning rate 1e-4 with cosine decay, effective batch size 32 (micro-batch 4), 3 epochs, cutoff 1024 tokens
- Generation with vLLM: temperature 0.2, top-k 10, maximum 500 new tokens
- GPT-4o mini is used only as a judge (relevance score) and to make golden labels for the document tracks

## How is this paper novel?

- No seed set at all. Instruction back-translation and its variants need labeled pairs to train the first backward model.
- Two models train each other. The pseudo-label of one model is the input of the other model.
- The training target is always real human text. Only the input side is generated.
- Cycle consistency is used two times: as the training signal, and as a data filter

## List of experiments

- Four data tracks: general instructions (Alpaca-GPT4, Dolly-15k), domain-specific (Medical-Alpaca), dialogue logs (OASST-1), plain text (WikiHow)
- For the instruction datasets, the pairs are broken: one side of each pair is removed to get unpaired questions and answers
- Baselines: Vanilla (base model), All-SFT (100% gold pairs), SFT-80 (80% gold pairs), Rand-k% and Clust-k% (seed-based back-translation with k = 5, 10, 20% seed data, random or k-means seed selection)
- Average accuracy on MMLU, BBH, CRASS, DROP:

| Data | Vanilla | Best seed-based | Cycle-Inst | Cycle-Filt | SFT-80 | All-SFT |
|---|---|---|---|---|---|---|
| Alpaca-GPT4 | 47.13 | 50.84 | 53.24 | 54.17 | 53.04 | 54.99 |
| Dolly-15k | 47.13 | 48.03 | 51.07 | 51.00 | 52.53 | 53.45 |
| OASST-1 | | 50.75 | 51.72 | 51.98 | 50.87 | 51.69 |
| WikiHow | | 49.63 | 50.62 | 51.07 | 50.59 | 50.97 |

- Medical-Alpaca (average on 8 MMLU medical sub-domains): Vanilla 0.589, best seed-based 0.632, Cycle-Inst 0.630, Cycle-Filt 0.636, SFT-80 0.625, All-SFT 0.626
- AlpacaEval win rate against All-SFT is reported in a figure only
- GPT-4o mini relevance score (0-10) of 500 generated pairs for Cycle-Inst: Alpaca 9.46, Dolly 9.90, MedAlpaca 9.96, OASST-1 8.75, WikiHow 9.43
- The relevance score correlates with downstream accuracy: Pearson r from 0.646 (MedAlpaca) to 0.904 (Alpaca)

### Ablation Studies

- There is no formal ablation table
- Filter on / off: the filter helps on Alpaca (53.24 -> 54.17), MedAlpaca (0.630 -> 0.636), OASST-1 (51.72 -> 51.98), WikiHow (50.62 -> 51.07). It does not help on Dolly (51.07 -> 51.00).
- Number of cycle iterations (1 -> 2 -> 3):
  - Alpaca-GPT4: 53.24 -> 54.01 -> 54.13
  - Dolly-15k: 51.07 -> 52.60 -> 53.13
  - OASST-1: 51.72 -> 51.20 -> 51.55
  - WikiHow: 50.62 -> 51.11 -> 51.15
  - Medical: 0.630 -> 0.632 -> 0.631
- Seed selection for the baselines: cluster-based seeds are more stable than random seeds

### Efficiency analysis

Not mentioned

## Preliminaries

### Instruction back-translation

- You have many good answers (for example web paragraphs) but no instructions for them
- Train a backward model `p(instruction | answer)` on a small seed set of human pairs
- Use the backward model to write an instruction for each unlabeled answer
- Fine-tune the forward model on `(generated instruction, real answer)`
- The idea comes from machine translation, where a target-to-source model makes synthetic source sentences

### Dual learning / cycle consistency

- Two tasks are dual if one is the inverse of the other: `f: X -> Y` and `g: Y -> X`
- Cycle consistency: `g(f(x))` must be equal to `x`, and `f(g(y))` must be equal to `y`
- This gives a training signal from unpaired data, because you do not need the true `y` for `x`. You only need `x` to come back.
- Known from CycleGAN (unpaired image translation) and unsupervised machine translation

### LoRA

- Freeze the weight `W`. Learn a low-rank update `W + B @ A`, where `A` and `B` are small matrices of rank `r`
- Only `A` and `B` are trained, so the fine-tune is cheap

## GPU hours

- Hardware: 8 x NVIDIA RTX 3090 (24 GB) for the instruction-corpus experiments, 8 x RTX 4090 (24 GB) for the raw-document experiments
- Training time / GPU hours: Not mentioned

## Key takeaways

- Two models that are inverses of each other can bootstrap from unpaired text with zero labeled pairs
- Keep the real text on the target side and the generated text on the input side. The model then never learns to output its own mistakes.
- With 0% labels, Cycle-Filt is better than back-translation with 5-20% seed data on all five datasets (average accuracy). Cycle-Inst without the filter is better on four; on MedAlpaca it is 0.630 against 0.632 for the best seed-based run.
- It is near full supervision on most datasets. Dolly is the exception (51.00 against 53.45 for All-SFT).
- One cycle gives most of the gain. More iterations help a little on some datasets and not on others.
- Reconstruction distance is a usable quality filter, but the gain is small
- The `?` heuristic is weak: it fails on text that has no explicit questions (the authors list this as a limitation)
- Only LoRA was tested

## What I still do not understand?

## Ideas to pursue

- **In common with the causal-consistency objective:** Both use a pair of inverse operations on raw corpus text and no labels. Here the pair is (question -> answer, answer -> question). In the idea the pair is ("Why?" -> cause, "Therefore?" -> original line). In both, the generated text is the input and the real corpus line is the training target (steps 7 and 9 here; the correct option in the idea).
- **What it lacks:** The cycle stays inside one passage and makes instruction-following data. It does not connect facts from different documents, and it has no causal direction (a question is not a cause of its answer). It uses two models and free generation as the reconstruction step. It has no distractor, so nothing stops a generic pseudo-answer that is compatible with many questions.
- **What to borrow:** (1) The rule "real text is always the target" - train next-token loss only on the original line, never on the model's own "Why?" answer. (2) The reconstruction-distance filter - drop the abductions from which the original line cannot be recovered before they enter the loss. (3) The experiment design - compare against a seed-based baseline at 5/10/20% labels, and check if 2-3 iterations of the cycle add something after the first.

## Similar papers

- [Deductive Closure Training of Language Models for Coherence, Accuracy, and Updatability](https://arxiv.org/abs/2401.08574)
- [RevThink: Reverse Thinking Makes LLMs Stronger Reasoners](https://arxiv.org/abs/2411.19865)
- [Reasoning to Learn from Latent Thoughts](https://arxiv.org/abs/2503.18866)
- [Time-Reversal Provides Unsupervised Feedback to LLMs](https://arxiv.org/abs/2412.02626)
- [Flip-Flop Consistency: Unsupervised Training for Robustness to Prompt Perturbations in LLMs](https://arxiv.org/abs/2510.14242)
- [LEDOM (ACL 2026)](https://aclanthology.org/2026.acl-long.975)
- [The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"](https://arxiv.org/abs/2309.12288)
- [Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations](https://arxiv.org/abs/2205.11822)
