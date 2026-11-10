# [Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations](https://arxiv.org/abs/2205.11822)

## Meta

* Journal   - EMNLP
* Year      - 2022
* Author    - Jaehun Jung, Lianhui Qin, Sean Welleck, Faeze Brahman, Chandra Bhagavatula, Ronan Le Bras, Yejin Choi (University of Washington, Allen Institute for AI)
* Code      - [github.com/jaehunjung1/Maieutic-Prompting](https://github.com/jaehunjung1/Maieutic-Prompting)
* Slides    - 
* One liner - At inference time, build a recursive tree of abductive explanations for and against a statement, then solve for the most consistent truth assignment with MAX-SAT.
* Model     - GPT-3 text-davinci-001, RoBERTa-MNLI verifier
* Datasets  - Com2Sense, CSQA 2.0, CREAK, StrategyQA (appendix)
* Baselines - Standard prompting, chain of thought, self-consistency, generated knowledge prompting, supervised RoBERTa-large / T5-11B / Unicorn-11B

## What problem does it solve?

- Explanation-based prompting (for example Chain of Thought) relies on explanations that the LM generates. These explanations are frequently noisy and inconsistent.
- The paper inspected 100 GPT-3 samples and found three types of logical error:
  - The explanation does not logically lead to the answer.
  - The model gives the same label to a statement and to its negation.
  - The model says its own generated explanation is false.
- The goal: get a correct True/False answer from explanations that are not reliable.

## How does it solve it?

There is no training. The method is inference-only (few-shot prompting plus a symbolic solver).

### Training flow

No training. The LM (GPT-3) and the NLI verifier (RoBERTa fine-tuned on MNLI) are used as they are.

### Inference

1. Start with a statement `Q`. The task is to decide if `Q` is True or False.
2. Abductive generation: prompt the LM with "`Q`? True, because" to get an explanation `E_T`. Prompt it with "`Q`? False, because" to get an explanation `E_F`.
3. Do a check of each explanation `E` for logical integrity: the LM must give opposite labels to `E` and to `not E`.
4. If `E` is logically integral, stop this branch.
5. If `E` is not logically integral, use `E` as a new question and generate `E_T` and `E_F` for it (recursion). Stop at the maximum depth.
6. Prune the branches whose leaf nodes are still not logically integral. The result is the maieutic tree.
7. Calculate a belief weight for each leaf node (how strongly the LM thinks the node is true).
8. Run an NLI model on all pairs of nodes. Each "entail" or "contradict" result becomes a logical clause with weight 1.
9. Give all clauses to a weighted MAX-SAT solver. The solver assigns True or False to all nodes and to `Q`.
10. The value of `Q` is the answer.

### Equations

#### Logical integrity

```python
def integral(E):
    a = argmax_A(p_lm(A | E, C))        # label for E
    b = argmax_A(p_lm(A | not_E, C))    # label for the negation of E
    return (a == True and b == False) or (a == False and b == True)
```

- `C` - the few-shot examples
- `not_E` - made with a prefix such as "It is wrong to say that", or the LM is prompted to negate the sentence
- A proposition is "logically integral / True" if the first case holds and "logically integral / False" if the second case holds

#### Abduction

```python
def abduction(Q):
    E_T = sample(p_lm(E | Q, True, C))    # "Q? True, because ..."
    E_F = sample(p_lm(E | Q, False, C))   # "Q? False, because ..."
    return E_T, E_F
```

#### Belief (unary weight)

`w_E = (p_lm(True | E, C) - p_lm(True | not_E, C)) / (p_lm(True | E, C) + p_lm(True | not_E, C))`

- This needs no additional LM calls. The values are known from the integrity check.
- Belief constraint for each leaf node: the clause is `E` if the node is integral / True, and `not E` if the node is integral / False.

#### Consistency (binary weight, first version)

`w_{E,Q,A} = p_lm(E | Q, A, C) / (p_lm(E | Q, A, C) + p_lm(E | Q, not_A, C))`

- If the LM is consistent, `E_F` must be more likely after "False" than after "True".
- Consistency constraint for each edge (parent `E_l`, child `E_lA`): `E_lA -> E_l` if `A = True`, and `E_lA -> not E_l` if `A = False`.

#### Verifier (final version of the consistency clauses)

- The first version only relates a parent and a child. It cannot relate nodes in different branches.
- Thus the paper uses an NLI model on all pairs of nodes `(E1, E2)`:
  - `Entail(E1, E2)` gives the clause `E1 -> E2`
  - `Contradict(E1, E2)` gives the clause `E1 -> not E2`
- The weight of each NLI clause is fixed at 1. The paper also tried the NLI label probability as the weight. Weight 1 was better.

#### Objective (weighted MAX-SAT)

```python
# find the True/False assignment to all nodes and Q that maximizes
score = sum([w[c] for c in belief_clauses + consistency_clauses if is_satisfied(c)])
```

### Model

- LM: GPT-3 `text-davinci-001`, with 6 few-shot examples
- Verifier: RoBERTa fine-tuned on MNLI (90.2% accuracy on the MNLI dev set)
- MAX-SAT solver: RC2
- Maximum tree depth: 2
- Depth 1: nucleus sampling (`p = 1.0`), 3 `E_T` and 3 `E_F`
- Depth 2: greedy decoding, 1 `E_T` and 1 `E_F` for each parent
- Maximum tree size: 18 nodes (without `Q`)

## How is this paper novel?

- Abductive generation: the LM must explain the two answers (True and False). Other methods generate one explanation and then predict the label.
- Recursive explanations: the LM validates its own explanations. This gives depth, not only width as in Self-Consistency.
- Logical integrity: a test to find the propositions for which the LM is self-consistent.
- The explanations are not used "as-is". The method models the relations between the explanations and solves them together with MAX-SAT.
- Fully unsupervised. It works with any LM that can be prompted few-shot.

## List of experiments

Accuracy of the prompting methods, all with the same GPT-3 model:

| Method | Com2Sense dev | CSQA 2.0 dev | CREAK dev | CREAK contrast |
|---|---|---|---|---|
| Standard | 58.1 | 54.1 | 60.3 | 55.2 |
| Chain of Thought | 61.6 | 59.6 | 64.8 | 59.4 |
| Self Consistency | 61.4 | 60.8 | 70.5 | 64.8 |
| GKP | 61.8 | 59.7 | 75.4 | 68.2 |
| Maieutic Prompting | 72.5 | 69.5 | 85.2 | 77.4 |

- Maieutic test results: Com2Sense test 75.0 and pairwise 68.7, CSQA 2.0 test 68.3, CREAK test 85.3.
- Supervised baselines for comparison: RoBERTa-large (Com2Sense dev 62.8, CREAK dev 80.6), T5-11B (Com2Sense dev 77.2, CSQA 2.0 dev 68.5, CREAK dev 89.5), Unicorn-11B (CSQA 2.0 dev 69.9, test 70.2).
- Self Consistency and GKP use N = 20 explanations. This is more than the maximum tree size of 18.
- Robustness to semantic changes: the pairwise accuracy (Com2Sense) and the contrast-set accuracy (CREAK) pair each question with an almost equal question that has the opposite answer. The gap between Maieutic and the baselines becomes larger on these metrics.
- Robustness to prompts: 3 different sets of few-shot examples and 5 different orders of the examples. Maieutic and Self Consistency are much more stable than Standard and Chain of Thought. Maieutic has a slightly smaller variance.
- Human evaluation: six NLP experts scored 100 CSQA 2.0 dev examples (50 correct answers, 50 incorrect answers). More than 99% of the "true Es" are grammatical. For the incorrectly answered examples, 42% of the true Es are fully factual and 23% are fully helpful. Krippendorff's alpha = 0.64.
- StrategyQA dev (appendix): Standard 56.3, Chain of Thought 58.2, Maieutic 60.7, Chain of Thought multi-hop 57.9, Maieutic multi-hop 61.4.

### Ablation Studies

Com2Sense dev accuracy (full method: 72.5):

- Non-abductive generation (no answer label in the prompt): 68.4
- All greedy decoding: 67.2
- All nucleus sampling: 72.0
- Likelihood-based consistency (no NLI verifier): 65.6

Tree size (Com2Sense dev):

| Dimension | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| Depth | 61.3 | 72.5 | 72.4 | - | - |
| Width | 62.4 | 66.5 | 72.5 | 71.5 | 72.1 |

- The accuracy saturates in the two dimensions.
- Reasons given: topic drift when the depth increases, and overlap of the generated knowledge when the width increases.

### Efficiency analysis

- No measurement of cost or number of LM calls is given.
- The paper says a full tree is expensive, because the number of generations grows exponentially with the depth. Thus generation stops in a branch when a logically integral proposition is found.

## Preliminaries

### Abduction

- Deduction: from a cause, find the effect. Abduction: from an observation, find the most plausible explanation.
- Here: the answer label is given first ("True, because ..."), and the LM generates a reason for it after the fact.
- This is different from Chain of Thought, where the explanation comes first and the label comes last.

### Weighted MAX-SAT

- SAT: given logical clauses on boolean variables, find an assignment that satisfies all clauses.
- MAX-SAT: if not all clauses can be satisfied, satisfy as many as possible.
- Weighted MAX-SAT: each clause has a weight. Maximize the sum of the weights of the satisfied clauses.

```python
best = max(all_assignments, key=lambda a: sum([w[c] for c in clauses if satisfied(c, a)]))
```

- An off-the-shelf solver (RC2) does this search.

### Natural language inference (NLI)

- Input: two sentences (premise, hypothesis).
- Output: entail, contradict, or neutral.
- Here: entail becomes the implication `E1 -> E2`, contradict becomes `E1 -> not E2`, neutral gives no clause.

## GPU hours

Not mentioned. The LM is used through the OpenAI GPT-3 API.

## Key takeaways

- An LM is frequently not consistent with itself: it can give the same label to a statement and to its negation.
- Do not use one explanation as-is. Generate many explanations for the two answers, then find the subset that is consistent.
- "Why is it true?" and "Why is it false?" asked recursively gives a tree. Recursion stops at propositions that the LM believes consistently.
- A small NLI model plus a symbolic solver can repair the inconsistency of a large LM at inference time.
- Depth 2 and width 3 are sufficient. More does not help.
- Limitations given by the authors: only True/False statements, and relations are modelled only in one tree, not between the trees of different questions.

## What I still do not understand?

## Ideas to pursue

- **In common:** The abductive step ("`Q`? True, because ...") is the same operation as the "Why?" step of the causal-consistency objective. The recursion (explain the explanation) is the same as repeated "why" that makes a graph. The stop condition (a logically integral proposition) is a possible definition of a "root cause": a node that the model believes consistently and that needs no more explanation. The consistency weight `p(E | Q, A)` against `p(E | Q, not A)` is a two-option contrast, the same form as the correct line against the distractor line.
- **What it lacks:** No training. The model stays inconsistent and the solver repairs each question again at inference time. The explanation direction is one-way: the paper never does a check that the explanation leads back to the question ("Therefore?"). The first version of the consistency weight only uses the likelihood of the same generation. The scope is one tree for one question. The authors say in the limitations that relations between trees (knowledge from one question used as evidence for a different question) are future work. This is the cross-document consistency that the idea targets.
- **What to borrow:** (1) The logical integrity test as a filter: keep only "Why?" answers for which the model gives opposite labels to the answer and its negation, before they are used as training input. (2) The NLI verifier to find relations between "why" answers of different lines, so that the graph has cross-branch edges. (3) Abduction for the distractor also: generate "why" answers for the distractor line to make hard negatives. (4) The MAX-SAT assignment as a source of pseudo-labels for the next-token loss.

## Similar papers

- [Deductive Closure Training](https://arxiv.org/abs/2401.08574)
- [RevThink](https://arxiv.org/abs/2411.19865)
- [Reasoning to Learn from Latent Thoughts](https://arxiv.org/abs/2503.18866)
- [Cycle-Instruct](https://arxiv.org/abs/2508.16100)
- [Time-Reversal Provides Unsupervised Feedback to LLMs](https://arxiv.org/abs/2412.02626)
- [Flip-Flop Consistency](https://arxiv.org/abs/2510.14242)
- [LEDOM: Reverse Language Model](https://aclanthology.org/2026.acl-long.975)
- [The Reversal Curse](https://arxiv.org/abs/2309.12288)
