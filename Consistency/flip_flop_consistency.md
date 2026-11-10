# [Flip-Flop Consistency: Unsupervised Training for Robustness to Prompt Perturbations in LLMs](https://arxiv.org/abs/2510.14242)

## Meta

* Journal   - arXiv preprint
* Year      - 2025
* Author    - University of Southern California (Parsa Hejabi, Elnaz Rahmati, Alireza S. Ziabari, Morteza Dehghani)
* Code      - [github](https://github.com/ParsaHejabi/Flip-Flop-Consistency-Unsupervised-Training-for-Robustness-to-Prompt-Perturbations-in-LLMs)
* Slides    - 
* One liner - Unsupervised training that pulls the outputs for low-confidence prompt variants toward the consensus of the confident variants, so the answer does not flip with the prompt wording.
* Model     - Qwen2.5-3B-Instruct (LoRA)
* Datasets  - ANLI R1/R2/R3, CB, RTE, COPA, HellaSwag, StoryCloze 2016, WSC, Winogrande-XL, WiC (prompt templates from PromptSource / P3)
* Baselines - Zhou et al. (2022) prompt-consistency loss

## What problem does it solve?

- An LLM often gives different answers to prompts that have the same meaning but different wording
- Prompt engineering adds cost and does not repair the inconsistency inside the model
- Supervised fine-tuning needs labels
- Swarm distillation (pairwise KL between prompt variations) needs no labels, but it can decrease task performance
- The paper wants consistency across prompt variations, with no labels, and without a loss in task performance

## How does it solve it?

### Training flow

1. Take an unlabeled example `i` and write it in `V` prompt variations (PromptSource templates, 4 to 15 for each dataset)
2. For each variation `v` and each candidate label `c`, compute the length-normalized log-likelihood of the label text
3. Each variation votes for its best label
4. If no label has a strict majority (`> V/2` votes), give this example no loss
5. The majority label is the consensus pseudo-label `c*`
6. Consensus Cross-Entropy (CCE): train all variations to output `c*`
7. Compute the margin of each variation that agrees with the consensus (score of `c*` minus score of the best other label)
8. If all variations agree and the median margin is above a threshold: add a JSD loss that pulls the output distributions of all variations together
9. If the vote is split: the top-k agreeing variations by margin are the consensus-confident (CC) set. The other variations are the non-consensus (NC) set.
10. Add a KL loss that pulls each NC variation toward the mean distribution of the CC set (the "flip"), and a JSD loss inside the CC set
11. The backbone is frozen. Only LoRA weights are trained.
12. Keep the checkpoint with the best mean F1 on the validation split

### Inference

1. Use the fine-tuned model as usual with one prompt
2. No voting and no extra calls at inference

### Equations

#### Scores and votes

```python
# length-normalized log-likelihood of label c under prompt variation v
LL[i][v][c] = mean([log_p(y[t], given=x[i][v] + y[:t]) for t in range(T)])
pi[i][v] = softmax(LL[i][v])            # over labels c
y_hat[i][v] = argmax(LL[i][v])          # vote of variation v

n[i][c] = sum([y_hat[i][v] == c for v in range(V)])   # vote count
c_star = argmax(n[i])                   # used only if n[i][c_star] > V / 2
```

#### Consensus Cross-Entropy

```python
L_CCE[i] = (n[i][c_star] > V / 2) * lambda_CCE * mean([nll(c_star, given=x[i][v]) for v in range(V)])
```

- All variations, also the ones that voted for a different label, are trained on the majority label

#### Alignment losses

```python
# q[i][v] = softmax over the full vocabulary at the consensus answer tokens
G = [v for v in range(V) if y_hat[i][v] == c_star]         # variations that agree
m[v] = LL[i][v][c_star] - max([LL[i][v][c] for c != c_star]) # margin

# Case 2: unanimous and confident (len(G) == V and median(m) >= tau_unanimous)
L_jsd[i] = beta_jsd * JSD([q[i][t] for t in G])

# Case 3: split vote
CC = top_k(G, by=m, k=min(k_max, V - 1))    # consensus-confident, k >= 2
NC = [v for v in range(V) if v not in CC]   # non-consensus
q_bar = mean([q[i][t] for t in CC])         # mean in probability space
L_flip[i] = w_flip * mean([KL(q[i][s], q_bar) for s in NC])
L_jsd[i] = beta_jsd * JSD([q[i][t] for t in CC])

# weight of the flip loss
delta = mean([LL[i][t][c_star] for t in CC]) - mean([LL[i][s][c_star] for s in NC])
w_flip = f_min + (f_max - f_min) * sigmoid(delta / t)
```

#### Total loss for one example

```python
if no_strict_majority:        L[i] = 0
elif unanimous_and_confident: L[i] = L_CCE[i] + L_jsd[i]
else:                         L[i] = L_CCE[i] + L_jsd[i] + L_flip[i]
```

- The paper does not give numeric values for `lambda_CCE`, `tau_unanimous`, `k_max`, `f_min`, `f_max`, `t`, `beta_jsd`
- The paper does not say if the gradient is stopped on the CC target `q_bar`

#### Metric: observed agreement

`P_i = (1 / (V * (V - 1))) * sum over c of n[i][c] * (n[i][c] - 1)`

- `P_o` is the mean of `P_i` over the examples
- It is the fraction of pairs of variations that give the same label

### Model

- Qwen2.5-3B-Instruct
- LoRA: rank 16, alpha 32, dropout 0.05, on the attention and MLP projections of each transformer block
- Learning rate, batch size, number of epochs: Not mentioned

## How is this paper novel?

- The pseudo-label is the majority vote of the model's own answers across prompt variations. No labels and no teacher.
- It does not align all pairs of variations equally (swarm distillation does that). It aligns the unsure variations toward the confident ones only.
- The strength of the alignment depends on the confidence gap between the CC set and the NC set
- Examples with no majority get no loss, so the model does not train on a pseudo-label that has no support

## List of experiments

- 11 classification datasets, 4 task types: NLI, sentence completion, coreference, word sense disambiguation
- Compared methods: Base (no training), Swarm (swarm distillation), CCE (only the consensus cross-entropy), F2C (full method)
- Metrics: mean F1 across variations, standard deviation of F1 across variations, observed agreement `P_o`
- Mean change against the base model:

| Method | P_o | mean F1 | std of F1 |
|---|---|---|---|
| F2C | +11.62% | +8.94% | -3.29% |
| CCE | +11.68% | +8.36% | -3.05% |
| Swarm | -0.38% | +1.40% | -0.47% |

- Selected rows (mean F1 / std F1 / P_o), Base -> F2C:
  - ANLI R1: 39.23 / 13.97 / 52.29 -> 58.43 / 2.61 / 78.85
  - HellaSwag: 35.43 / 2.20 / 50.89 -> 70.28 / 2.12 / 76.97
  - StoryCloze: 85.35 / 7.78 / 85.24 -> 96.32 / 0.46 / 97.32
  - RTE: 77.90 / 4.12 / 85.08 -> 81.55 / 0.82 / 93.40
  - WiC: 11.45 / 17.53 / 85.00 -> 9.99 / 16.66 / 87.01
- F1 improves on 9 of 11 datasets. On CB, WSC and WiC the base model is weak and has high variance, so the majority vote is not reliable. Agreement still increases there, but F1 does not.
- **Out-of-domain:** train on one dataset, test on the others (80 source -> target pairs). Mean change against base: F1 +7.613 (74 of 80 pairs positive), `P_o` +7.491 (64 of 80 positive), std of F1 decreases by 2.941 (66 of 80 pairs).
- **Unseen prompt formats:** train on a subset of the formats (for example 5 of 10 on RTE, 5 or 10 of 15 on ANLI) and test on the held-out formats. F1 and agreement improve on the held-out formats, and the variance decreases as the number of training formats increases. Exact numbers are in a figure only.

### Ablation Studies

- One ablation only: CCE alone against full F2C
- The alignment losses add a small gain over CCE: mean F1 +8.36% -> +8.94%, std of F1 -3.05% -> -3.29%. Observed agreement is almost the same (+11.68% against +11.62%).
- No ablation of the hyper-parameters (thresholds, `k_max`, weights). The authors say these were not fully tuned.

### Efficiency analysis

- Training time is approximately 3 to 23 hours for each dataset. It depends on the dataset size and the number of prompt formats.
- No other efficiency analysis

## Preliminaries

### KL divergence and Jensen-Shannon divergence

```python
KL(p, q) = sum([p[x] * log(p[x] / q[x]) for x in vocab])   # not symmetric

# JSD of a set of distributions: mean KL of each one to their average
m = mean(ps)
JSD(ps) = mean([KL(p, m) for p in ps])
```

- KL pulls one distribution toward another
- JSD is symmetric and bounded. It pulls a group of distributions toward their common average.

### Pseudo-labeling and consistency regularization

- Pseudo-labeling: use the model's own confident prediction on unlabeled data as the label, then train on it
- Consistency regularization: the prediction must not change when the input is changed in a way that keeps the meaning
- F2C uses the two together: the views are prompt templates, and the pseudo-label is the majority vote over the views

### Swarm distillation

- A baseline from Zhou et al. (2022)
- It minimizes the KL divergence between all pairs of prompt variations of the same input
- It has no notion of which variation is correct, so the model can become consistent on a wrong answer

## GPU hours

- Small datasets: a pool of seven NVIDIA RTX A6000 GPUs (48 GB each)
- Large datasets: one NVIDIA A100 GPU (80 GB)
- Total training time: approximately 3 to 23 hours for each dataset
- Exact GPU hours: Not mentioned

## Key takeaways

- A majority vote across paraphrased prompts is a good unsupervised label if the base model is already acceptable on the task
- Most of the gain comes from the simple consensus cross-entropy. The more complex alignment losses add little.
- Pairwise alignment with no notion of correctness (swarm) does almost nothing here
- If the base model is weak and noisy on a task, the method makes the model more consistent but not more correct
- The consistency transfers to other datasets and to prompt formats that were not seen in training
- Limits: one 3B model, classification only, template-based perturbations only

## What I still do not understand?

## Ideas to pursue

- **In common with the causal-consistency objective:** Both are unsupervised consistency losses in which the model's own outputs make the training signal, and both end in a next-token / cross-entropy loss on a chosen option among candidate labels. Both need a rule to decide which self-generated signal to trust.
- **What it lacks:** The consistency is across paraphrases of the same input (an invariance), not across a pair of inverse operations. There is no cycle, no abduction, no causal direction, and no link between different documents. It needs a set of hand-written templates for each task and works only on classification with a fixed label set.
- **What to borrow:** (1) The gate - give no loss when there is no strict majority. In the idea: sample several "Why?" answers for one line and train only if most of them let the model recover the original line. (2) The confidence margin (score of the correct option minus score of the best distractor) as a weight for the loss or as a filter for weak abductions. (3) The metrics and the warning - report agreement and variance apart from accuracy, because the method can make a weak model consistently wrong (CB, WSC, WiC).

## Similar papers

- [Deductive Closure Training of Language Models for Coherence, Accuracy, and Updatability](https://arxiv.org/abs/2401.08574)
- [RevThink: Reverse Thinking Makes LLMs Stronger Reasoners](https://arxiv.org/abs/2411.19865)
- [Reasoning to Learn from Latent Thoughts](https://arxiv.org/abs/2503.18866)
- [Cycle-Instruct: Fully Seed-Free Instruction Tuning via Dual Self-Training and Cycle Consistency](https://arxiv.org/abs/2508.16100)
- [Time-Reversal Provides Unsupervised Feedback to LLMs](https://arxiv.org/abs/2412.02626)
- [LEDOM (ACL 2026)](https://aclanthology.org/2026.acl-long.975)
- [The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"](https://arxiv.org/abs/2309.12288)
- [Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations](https://arxiv.org/abs/2205.11822)
