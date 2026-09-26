# [Training Large Language Models to Reason in a Continuous Latent Space (Coconut)](https://arxiv.org/abs/2412.06769)

**TL;DR:** Instead of decoding each reasoning step into a word, feed the last hidden state straight back in as the next input ("continuous thought"). Trained with a curriculum that swaps language steps for continuous thoughts, GPT-2 learns latent reasoning that holds several paths at once, like breadth-first search, and beats language CoT on planning-heavy logic.

**Source Code:** [facebookresearch/coconut](https://github.com/facebookresearch/coconut)

**Datasets:** GSM8K, ProntoQA, ProsQA (new: logical reasoning over DAGs)

**Author:** Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian (Meta FAIR, UC San Diego)

**Journal:** arXiv

**Year of Submission:** 2024

**Youtube:**

## What problem does it solve?

- Language CoT forces every intermediate step through a single sampled token, so each step has to commit to one path
- [Stepwise Internalization](stepwise_internalization.md) removes the steps entirely. Can the steps instead be kept, **but not as words**?
- This sits between a verbal and a non-verbal policy: the reasoning still happens step by step, but in vectors

## How does it solve it?

### Continuous thought

- In latent mode, between `<bot>` and `<eot>`, the model's **last hidden state is used directly as the next input embedding**, with no decoding to a token
- Outside latent mode, generation is normal

### Curriculum (follows iCoT)

- Stage 0: normal language CoT
- Stage k: replace the first `k` language reasoning steps with `k × c` continuous thoughts (c continuous thoughts per step)
- Loss on the remaining language steps and the answer. The question and the continuous thoughts are masked, so the thoughts are trained to help *predict the future reasoning*, not to reconstruct the removed words.

### Model

GPT-2 (plus some Llama 3.2 3B / Llama 3 8B experiments)

### Pseudocode

```python
def generate_with_continuous_thoughts(model, question, num_thoughts):
    input_embeddings = model.embed(question + ["<bot>"])
    for _ in range(num_thoughts):
        last_hidden_state = model(inputs_embeds=input_embeddings).hidden_states[-1][:, -1:]
        input_embeddings = torch.cat([input_embeddings, last_hidden_state], dim=1)  # no token is decoded
    input_embeddings = torch.cat([input_embeddings, model.embed(["<eot>"])], dim=1)
    return model.generate(inputs_embeds=input_embeddings)  # answer in language


def stage_k_example(question, reasoning_steps, answer, stage, thoughts_per_step):
    num_thoughts = stage * thoughts_per_step
    remaining_steps = reasoning_steps[stage:]  # the first `stage` steps become continuous thoughts
    return question, num_thoughts, remaining_steps + [answer]


for stage in range(max_stage + 1):
    for question, reasoning_steps, answer in DATA:
        question, num_thoughts, target_text = stage_k_example(question, reasoning_steps, answer, stage, THOUGHTS_PER_STEP)
        # loss only on target_text; question and continuous thoughts are masked
        loss = -model.log_prob_with_continuous_thoughts(target_text, given=question, num_thoughts=num_thoughts)
        loss.backward()
```

## How is this paper novel?

- Reasoning in the **unrestricted hidden-state space** rather than the vocabulary
- Continuous thoughts can **superpose several candidate next steps**, which gives search-like reasoning a single language CoT can't express

## List of experiments

| Method | GSM8K | ProntoQA | ProsQA |
|---|---|---|---|
| Language CoT | **42.9%** | 98.8% | 77.5% |
| No CoT | 16.5% | 93.8% | 76.7% |
| iCoT | 30.0% | 99.8% | **98.2%** |
| **Coconut** | 34.1% | **99.8%** | 97.0% |

- GSM8K: 8.2 generated tokens vs 25.0 for CoT, keeping 79.5% of CoT's accuracy
- ProsQA (planning): Coconut beats language CoT by about 20 points with far fewer tokens

### Analysis: what the latent reasoning looks like

- Forcing a continuous thought to decode into language shows a **distribution over several candidate next nodes**. The latent reasoning keeps multiple frontier nodes alive, like BFS, and prunes them over later thoughts.
- The probabilities act as an implicit **value function**. Nodes closer to the leaves get sharper values, so delaying commitment helps planning.

### Ablation Studies

- **Without the language-CoT curriculum, Coconut is no better than No-CoT.** The latent reasoning has to be bootstrapped from verbal reasoning.
- c = 3 thoughts per step causes training loss spikes on GSM8K

### Efficiency analysis

- Fewer generated tokens, but training needs sequential forward passes per thought, which is hard to parallelize

## Preliminaries

### Superposition in latent reasoning

A token commits to one discrete choice; a hidden-state vector can encode a weighted mix of several. Language CoT is like depth-first commitment, and continuous thought allows breadth-first exploration.

## GPU hours

GPT-2 scale, plus some 3B–8B runs.

## Key takeaways

- A verbal policy can be moved into a **latent-but-still-sequential** policy, and for planning tasks the latent one can be *better* than the verbal one it came from
- The non-verbal version **needs the verbal one first** (the curriculum ablation). Words are the scaffold for learning latent reasoning.
- The latent policy is only **partly re-verbalizable**: decoding a thought gives a *distribution* over steps rather than one step. A single language CoT can't faithfully articulate a superposed search state.
- Gains are smaller on larger pretrained models, perhaps because they are strongly biased toward reasoning in language

## Flipping the direction: verbalizing continuous thoughts

*My proposal, extending the paper's own decoding analysis.*

**Verdict: feasible, but only with a richer verbal format than a single chain.**

**The paper already does half of it.** Its probe decodes each continuous thought into language and gets a *distribution* over candidate next nodes. That is a verbalization of the latent policy.

**The obstacle is superposition.** A continuous thought can hold several paths at once. A language step commits to one, so a plain one-path CoT is a *lossy* articulation. The paper's own numbers show the loss: language CoT gets 77.5% on ProsQA, Coconut gets 97.0%. A faithful articulation has to be able to say the superposed state, e.g. "frontier: A (0.6), B (0.3), C (0.1)", instead of picking one branch.

**Reverse curriculum:** replace continuous thoughts with verbalized frontier steps one at a time, starting from the last thought (where the distribution is sharpest, according to the paper's value analysis).

**Faithfulness check:** re-run the model on the verbalized steps alone, with no continuous thoughts. If accuracy holds, the verbal version carries the policy. If it drops toward language-CoT levels, the single-path format is losing information.

```python
def verbalize_continuous_thought(model, thought_vector, top_k=3):
    # The paper's probe: decode a continuous thought into a distribution over next steps
    probabilities = softmax(model.unembed(thought_vector))
    top_candidates = top_k_with_probabilities(probabilities, k=top_k)
    return "frontier: " + ", ".join(f"{candidate} ({probability:.2f})" for candidate, probability in top_candidates)


def articulate_latent_reasoning(model, question, num_thoughts):
    thought_vectors = model.continuous_thoughts(question, num_thoughts)
    return [verbalize_continuous_thought(model, thought_vector) for thought_vector in thought_vectors]


def articulation_preserves_policy(model, question, num_thoughts):
    latent_answer = model.generate_with_continuous_thoughts(question, num_thoughts)
    verbal_steps = articulate_latent_reasoning(model, question, num_thoughts)
    verbal_answer = model.generate(question + "\n".join(verbal_steps))  # language only, no latent thoughts
    return verbal_answer == latent_answer


# Reverse curriculum: swap continuous thoughts back to verbal frontier steps, last thought first
for stage in range(1, num_thoughts + 1):
    for question, answer in DATA:
        verbal_steps = articulate_latent_reasoning(frozen_coconut_model, question, num_thoughts)
        num_latent = num_thoughts - stage
        loss = -model.log_prob_with_continuous_thoughts(
            verbal_steps[num_latent:] + [answer], given=question, num_thoughts=num_latent
        )
        loss.backward()
```

## What I still do not understand?

## Ideas to pursue

- After Coconut training, ask the model for a language CoT. Does it pick the top branch from its superposed state, or produce something else? That would be a direct test of how a non-verbal policy gets articulated.

## Similar papers

- [Stepwise Internalization](stepwise_internalization.md) (the curriculum it adapts)
- Shen et al. 2025, [*CODI*](https://arxiv.org/abs/2502.21074) (continuous thoughts learned by self-distillation instead of a curriculum)
- Goyal et al. 2023, *Think before you speak: pause tokens*; Pfau et al. 2024, *Let's Think Dot by Dot* (hidden computation in filler tokens)
