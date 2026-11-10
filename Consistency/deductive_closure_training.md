# [Deductive Closure Training of Language Models for Coherence, Accuracy, and Updatability](https://arxiv.org/abs/2401.08574)

## Meta

* Journal   - Findings of ACL
* Year      - 2024
* Author    - Boston University, MIT, IBM Research, Monash University Indonesia (Akyürek, Akyürek, Choshen, Wijaya, Andreas)
* Code      - [project page](https://lingo-mit.github.io/deductive-closure) (the paper says the code "will be made publicly available")
* Slides    - 
* One liner - The LM writes implications and contradictions of seed statements, picks the most probable logically consistent truth assignment, and fine-tunes on the statements marked true.
* Model     - Llama-2-7B (LoRA)
* Datasets  - CREAK, CREAK contrast set, MQuAKE-CF, Reversal Curse
* Baselines - Prompting, fine-tuning on seeds, MeLLo, MEMIT, graph inference

## What problem does it solve?

- An LM can give a truth value to one claim, but it does not keep one coherent model of the world
- The LM writes text that is wrong or that contradicts its other text
- The LM is hard to edit: a new fact goes in, but the consequences of the fact do not
- The usual training objective raises the probability of the training sentence only. It does not raise the probability of the logical consequences of that sentence
- Target property is deductive closure: if the LM gives high probability to `P`, and `P` implies `Q`, then the LM must also give high probability to `Q`. All accepted statements must agree with each other

## How does it solve it?

### Training flow

1. Get seed documents `s_i`. Three sources are possible: the LM itself (unsupervised), a trusted dataset (supervised), or a new fact / edit from the user (model updating)
2. Prompt the same LM with `p_imp` to write implications of the seed. These are statements that the seed entails
3. Prompt the same LM with `p_con` to write contradictions of the seed. Each prompt has 1-5 in-context examples
4. Make the related set `R_i = implications + contradictions + the seed`. The seed is in the set, so an error in the seed can also be found
5. Optional double-check: ask the LM if the seed really entails / contradicts each generated statement. Discard statements that do not get a high-probability "yes"
6. Ask the LM for the probability that each statement in `R_i` is true (few-shot prompt, read the normalized logits of the tokens "true" and "false")
7. Look at all truth assignments for `R_i`. Keep only the assignments that are logically consistent. Select the consistent assignment with the highest probability
8. Fine-tune the LM on the statements that the selected assignment marks as true
9. For fact verification, a statement marked false also gives supervision: the training text is "Verify the following statement: [claim]. False"

Variants

- **Unsupervised:** seeds are sampled from the LM. The consistency step can change the label of the seed and of the generated statements
- **Supervised:** seeds come from trusted data. The truth value of the seed is fixed to 1
- **Semi-supervised:** pool the generated data of the unsupervised run and the supervised run
- **Model updating:** the edit is the seed with truth value fixed to 1. Fine-tune only on the seed and its implications. Discard generated text that is not consistent with the seed
- **Transductive:** use the unlabeled evaluation claims to make seeds during training
- **Correlative implications (MQuAKE):** the model first writes related facts, then writes the implications

### Inference

1. No change at inference time. Use the fine-tuned LM as a normal LM
2. The reasoning (implications, contradictions, consistency check) is done only at training time

### Equations

#### Related set

```python
I[i] = [sample(p_LM, prompt=p_imp, seed=s[i]) for _ in range(n)]  # implications
C[i] = [sample(p_LM, prompt=p_con, seed=s[i]) for _ in range(n)]  # contradictions
R[i] = I[i] + C[i] + [s[i]]
```

#### Probability of a truth assignment

`p(T_i | R_i) = Π_j p_LM(t_ij | r_ij)`

- `r_ij` - statement j related to seed i
- `t_ij` - truth value (0 or 1) of `r_ij`
- `T_i` - one full assignment of truth values for the set `R_i`
- Each statement is scored independently by the LM

#### Consistency indicator

`c(T_i) = Π_{j in I_i} 1[t_i → t_ij] · Π_{j in C_i} 1[t_i → ¬t_ij]`

```python
def implies(a, b):
    return (not a) or b

def consistent(t_seed, t_imp, t_con):
    ok_imp = all(implies(t_seed, t) for t in t_imp)      # seed true -> implication true
    ok_con = all(implies(t_seed, not t) for t in t_con)  # seed true -> contradiction false
    return ok_imp and ok_con
```

- `t_i` - truth value of the seed
- If the seed is false, then all assignments of the other statements are consistent

#### Select the best assignment

`T_i* = argmax_T c(T | R_i) · p(T | R_i)`

- Example in the paper: one seed + one implication + one contradiction gives 8 assignments. 4 of them are consistent

#### Fine-tuning objective

`argmax_θ Σ_{i,j} t_ij · log p_LM(r_ij)`

- Standard LM log-likelihood, but only on the statements with `t_ij = 1`
- Training runs for a fixed number of iterations

#### Proposition 1

- Covers a minimal unsupervised DCT: one implication for each seed, no consistency check, training to convergence
- Assumption 1: given a generated question `q`, the answer to the seed question `q_0` is correct with probability at least `p*`
- Assumption 2: conditioning on a correct `(q_0, a_0)` pair raises the probability of a correct answer by at least `1/p*`
- Result: `p_DCT(a* | q) > p_LM(a* | q)`

### Model

- Llama-2-7B. The same model is the generator, the evaluator and the fine-tuned model
- LoRA: rank 8, alpha 32, dropout 0.1
- AdamW, learning rate 1e-4, batch size 4, linear schedule, 100 warmup steps
- 30 epochs by default. 60 epochs for the supervised experiments. 1 epoch for the transductive setting
- Generation: temperature 0.6, top-p 0.9. Unsupervised seed generation: temperature 0.9
- No development set was used to tune the hyperparameters

## How is this paper novel?

- The LM uses its own inference-time reasoning to make the supervision for its own fine-tuning
- One procedure covers three use cases: unsupervised improvement, supervised alignment, and model updating
- The seed is also in the consistency check, so the procedure can find and correct an error in the seed
- Gives a formal condition under which unsupervised DCT improves accuracy (Proposition 1)
- Argues that some factual errors of LMs come from the training algorithm, not from the training data

## List of experiments

- **Fact verification (CREAK validation accuracy, mean of 3 seeds)**
  - Unsupervised: prompting 71.7, DCT seed only 80.0, DCT (Imp. + Cont.) 83.5, with double-check 83.7
  - Supervised (20 labeled claims): fine-tuning 77.2, DCT 80.6, DCT + double-check 81.7
  - Semi-supervised: 84.9
  - Transductive: graph-inference baseline 77.7, DCT (Rel.) 84.5, DCT (Rel.) + (Imp. + Cont.) + double-check 85.5
- **Coherence (CREAK contrast set, 250 pairs with opposite truth values, unsupervised)**
  - "Both True" (lower is better): prompting 34.4, DCT seed only 20.4, DCT contradictions only 12.0, DCT (Imp. + Cont.) 19.2
  - "Both Correct" (higher is better): prompting 36.8, DCT (Imp. + Cont.) 49.6
- **Model updating (MQuAKE-CF, multi-hop accuracy at 10 / 20 / 50 / 100 / 1000 edits)**
  - MeLLo: 15.3 / 18.3 / 12.0 / 12.1 / 11.4
  - Fine-tuning on edits: 0.7 / 9.0 / 5.5 / 6.7 / 4.1
  - MEMIT: 11.1 / 11.7 / 6.0 / 1.1 / 0.6
  - DCT (Imp.): 20.0 / 20.5 / 14.0 / 10.7 / 8.1
  - DCT (Corr. Imp.): 41.7 / 29.4 / 25.6 / 14.2 / 12.3
  - DCT (Corr. Imp. + Imp.): 30.0 / 35.6 / 15.6 / 18.9 / 15.4
- **Reversal Curse (exact match on the reverse direction)**
  - Child-to-Parent: fine-tuning 2.2, DCT (Imp.) 48.3
  - Person-to-Description: fine-tuning 3.7, DCT (Imp.) 10.7
  - Description-to-Person: fine-tuning 3.0, DCT (Imp.) 15.7
  - The paper says the problem is far from solved

### Ablation Studies

- **Remove the consistency evaluation (unsupervised CREAK):** 83.5 -> 77.5
- **Seed only vs. full DCT:** 80.0 vs. 83.5
- **Double-check:** about the same accuracy with fewer examples (unsupervised: 586 -> 313 statements, 83.5 -> 83.7; supervised: 40 -> 14 statements, 80.6 -> 81.7). In the transductive setting it gives 80.3 -> 85.5
- **Transductive, relations only vs. plus (Imp. + Cont.) without double-check:** 84.5 vs. 80.3. More generated data without the filter decreased accuracy
- **Type of implication (MQuAKE):** correlative implications are better than plain implications at most edit counts
- **Manual analysis of about 350 generations:**
  - 74 of 100 generated implications / contradictions were valid. After double-check, 33 were left and 27 of them were valid
  - On MQuAKE, only 30% (Imp.) and 36% (Corr. Imp.) of the generations were strict logical implications
  - On Reversal Curse, most implications were paraphrases or trivial. The model reversed 22% of "X's parent is Y" statements

### Efficiency analysis

Not mentioned

## Preliminaries

### Deductive closure

A set of statements is deductively closed if it contains all statements that logically follow from it.

- If `P` is in the set and `P → Q`, then `Q` must also be in the set
- For an LM: "in the set" means "has high probability"
- A closed and consistent set cannot contain both `Q` and `not Q`

### Material implication

`a → b` is false only when `a` is true and `b` is false.

```python
implies = lambda a, b: (not a) or b
```

- If the seed is false, the implication constraint is always satisfied. This is why DCT can mark a seed as false and still keep a consistent assignment

### Model editing

Change one fact in an LM without full retraining. Example: "The prime minister of X is now Y."

- Methods such as MEMIT change the weights directly
- A good edit must also change the answers to multi-hop questions that depend on the fact. MQuAKE measures this

## GPU hours

Not mentioned

## Key takeaways

- The consistency check is the important part. Without it, self-generated data is less useful (83.5 -> 77.5)
- An LM can improve its accuracy with no labels: generate, check consistency, fine-tune
- Training on implications of an edit is better than training on the edit alone for multi-hop questions
- Most generated "implications" are not strict logical implications. Many are paraphrases or plausible related facts
- Only one layer of implications is used. Multi-hop graphs are not tested
- Only English datasets and only one model (Llama-2-7B)

## What I still do not understand?

## Ideas to pursue

- **In common with the causal-consistency objective:** both are self-supervised post-training steps. The model writes related text from a seed (DCT: implications; the idea: the answer to "Why?"), then the loss is plain next-token prediction on text that passed a consistency test. Both want consistency across statements, not only high probability for single statements
- **What DCT lacks:** it has no inverse direction. DCT goes seed -> implication (deduction) and stops. It does not go back from the implication to the seed, so there is no cycle. It has no abduction step ("Why?"), no distractor / multiple-choice loss, and the seeds are short factual claims, not random lines from the pre-training corpus
- **What the idea can borrow:**
  - The consistency filter before the loss. Score the truth of the abduced "Why?" answer with the model and discard low-probability or inconsistent answers. DCT shows that training on unfiltered self-generated text is much worse
  - Contradictions as a source of distractors. Use `p_con`-style generation to make a hard distractor line for the "Therefore?" step
  - Keep the original line in the consistency set, as DCT does with the seed. Then a wrong corpus line can be found, not blindly reinforced
  - The Reversal Curse benchmark as an evaluation for the cycle loss

## Similar papers

- [RevThink: Reverse Thinking Makes LLMs Stronger Reasoners](https://arxiv.org/abs/2411.19865)
- [Reasoning to Learn from Latent Thoughts](https://arxiv.org/abs/2503.18866)
- [Cycle-Instruct](https://arxiv.org/abs/2508.16100)
- [Time-Reversal Provides Unsupervised Feedback to LLMs](https://arxiv.org/abs/2412.02626)
- [Flip-Flop Consistency](https://arxiv.org/abs/2510.14242)
- [LEDOM](https://aclanthology.org/2026.acl-long.975)
- [The Reversal Curse](https://arxiv.org/abs/2309.12288)
- [Maieutic Prompting](https://arxiv.org/abs/2205.11822)
