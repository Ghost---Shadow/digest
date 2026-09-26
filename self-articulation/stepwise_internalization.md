# [From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step](https://arxiv.org/abs/2405.14838)

**TL;DR:** Start from a model finetuned on explicit CoT and delete CoT tokens from the front a few at a time while continuing to finetune. The model gradually absorbs the removed steps into its forward pass: GPT-2 Small ends up solving 9×9 multiplication at 99% with no CoT.

**Source Code:** [da03/Internalize_CoT_Step_by_Step](https://github.com/da03/Internalize_CoT_Step_by_Step)

**Datasets:** 4×4 to 9×9 multiplication, GSM8K

**Author:** Yuntian Deng, Yejin Choi, Stuart Shieber (AI2, Waterloo, UW, Harvard)

**Journal:** arXiv

**Year of Submission:** 2024

**Youtube:**

## What problem does it solve?

- [Implicit CoT via KD](implicit_cot_kd.md) needs a teacher, an emulator and a three-stage pipeline, and it breaks on harder tasks (10% on 5×5 for the KD variant as re-run here)
- Is there a simpler way to move a verbal policy into the weights?

## How does it solve it?

### Stepwise Internalization (ICoT-SI)

1. Finetune on `question → full CoT → answer`
2. Each epoch, **remove more CoT tokens from the start** of the CoT, with a linear schedule: `tokens_removed(t) = floor(Δ · t / T)`, where Δ is the number of tokens removed per epoch (typically 8)
3. Keep finetuning on what remains. The model has to compute the removed steps internally to predict the rest.
4. End with `question → answer`: no CoT at all

### Stabilization tricks

- **Removal smoothing:** add a random offset `o ~ Exponential(λ)` (λ ≈ 4) to the number of removed tokens, so the model sometimes sees the next stage early
- **Optimizer reset:** reset AdamW's state every time the removal count changes, because stale second-moment estimates cause spikes

### Models

GPT-2 Small / Medium, Phi-3 3.8B, Mistral 7B

### Pseudocode

```python
TOKENS_REMOVED_PER_EPOCH = 8
SMOOTHING_RATE = 4


def truncate_chain_of_thought(chain_of_thought_tokens, num_removed):
    return chain_of_thought_tokens[num_removed:]  # drop steps from the front


model = finetune(base_model, [(question, chain_of_thought + answer) for question, chain_of_thought, answer in DATA])
previous_num_removed = 0

for epoch in range(num_epochs):
    scheduled_num_removed = TOKENS_REMOVED_PER_EPOCH * epoch
    if scheduled_num_removed != previous_num_removed:
        optimizer = AdamW(model.parameters())  # reset optimizer state at each new stage
        previous_num_removed = scheduled_num_removed

    for question, chain_of_thought, answer in DATA:
        random_offset = int(np.random.exponential(SMOOTHING_RATE))  # removal smoothing
        remaining_steps = truncate_chain_of_thought(chain_of_thought, scheduled_num_removed + random_offset)
        loss = -model.log_prob(remaining_steps + answer, given=question)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# After the final stage the CoT is empty: the model maps question -> answer in one pass
answer = model.generate(question)
```

## How is this paper novel?

- Internalization by **curriculum alone**: no teacher, no hidden-state distillation, just gradually taking the words away
- Extends implicit reasoning to 9×9 multiplication and to 7B models

## List of experiments

| Task | Model | No CoT | Explicit CoT | ICoT-SI (no CoT tokens) |
|---|---|---|---|---|
| 9×9 multiplication | GPT-2 Small | low | high | **99%** |
| GSM8K | Mistral 7B | 38% | 68% | **51%** |

- ICoT-SI reaches 95% on 5×5 multiplication where ICoT-KD reaches 10%
- It runs at No-CoT speed; explicit CoT on GSM8K is about 11× slower

### Ablation Studies

- Removing smoothing or the optimizer reset hurts stability
- Removing tokens too aggressively (Δ = 16) causes training to fail to converge

### Efficiency analysis

- Training is expensive: every removal stage needs more finetuning, and longer CoTs need more stages

## Preliminaries

### Curriculum learning

Train on easy versions of a task first and make it gradually harder. Here "harder" means fewer reasoning tokens to lean on.

## GPU hours

## Key takeaways

- A verbal policy can be internalized **one step at a time**. Each removed step becomes a computation the forward pass has to do.
- It works nearly perfectly on algorithmic tasks, but a gap remains on natural math (51% vs 68%)
- The authors note the model "loses interpretable intermediate steps" and **do not test** whether it can still produce or articulate the removed CoT

## Flipping the direction: stepwise externalization

*My proposal, not in the paper.*

**Verdict: feasible, if two things are added: a source of candidate CoTs, and pressure that makes the CoT load-bearing.**

**The naive flip is trivial and says nothing.** If the model was internalized from known CoTs, reversing the curriculum (adding the removed steps back) just retrains on text you supplied. That isn't the model articulating anything.

**The interesting case is a policy learned with no CoT at all** (e.g. answer-only SFT, as in [Reasoning-Trace Collapse](reasoning_trace_collapse.md)). The flip needs:

1. **Candidate CoTs from the model itself.** Sample CoTs from the model with reasoning on, and keep those whose final answer matches the model's own *non-verbal* answer. This is like STaR's rationalization step, but the target is the model's own policy rather than the gold answer.
2. **A reverse curriculum.** Add CoT steps back gradually, last step first (the ones closest to the answer), so each stage adds a little more verbal reasoning in front of the answer.
3. **Pressure that makes the CoT the policy.** Without it, the model already computes the answer internally and can learn to emit a CoT it ignores, which is rationalization by construction. One theoretically clean fix: **mask attention from the answer tokens to the question**. The answer can then only see the CoT, so the CoT must carry everything the answer needs. The CoT is the policy by construction. Check it afterwards with the causal tests (corrupt a step, and the answer should change).

```python
def self_consistent_cots(frozen_policy, question, num_samples=16):
    # Keep only CoTs that end where the NON-VERBAL policy ends
    nonverbal_answer = frozen_policy.generate(question, reasoning="off")
    samples = [frozen_policy.generate(question, reasoning="on") for _ in range(num_samples)]
    return nonverbal_answer, [sample.chain_of_thought for sample in samples if sample.answer == nonverbal_answer]


def answer_cannot_attend_to_question(question_length, cot_length, answer_length):
    # Causal mask, plus: answer positions may not attend to question positions.
    # The only route from question to answer is through the CoT.
    mask = causal_mask(question_length + cot_length + answer_length)
    answer_start = question_length + cot_length
    mask[answer_start:, :question_length] = 0
    return mask


frozen_policy = copy_and_freeze(model)
for stage in range(1, max_num_steps + 1):  # externalize: last step first
    for question in QUESTIONS:
        nonverbal_answer, candidate_cots = self_consistent_cots(frozen_policy, question)
        for chain_of_thought in candidate_cots:
            visible_steps = split_steps(chain_of_thought)[-stage:]
            mask = answer_cannot_attend_to_question(len(question), len(visible_steps), len(nonverbal_answer))
            loss = -model.log_prob(visible_steps + nonverbal_answer, given=question, attention_mask=mask)
            loss.backward()


def cot_is_load_bearing(model, question):
    output = model.generate(question, reasoning="on")
    corrupted_answer = model.generate(question, forced_chain_of_thought=corrupt_steps(output.chain_of_thought))
    return corrupted_answer != output.answer
```

**Open risk:** with the attention bottleneck, the model might learn to *encode* the answer in the CoT without explaining it (steganography, or just stating the answer early). The CoT would be load-bearing but not an articulation. Checking that the CoT's steps are individually meaningful, and that a different model can follow them, would still be needed.

## What I still do not understand?

## Ideas to pursue

- **Reverse the curriculum:** after full internalization, ask the model for a CoT again. Does it regenerate the original steps? If it does, is the answer still computed internally, which would make that CoT post-hoc in the sense discussed in [Thought Crime](thought_crime.md)?
- Probe the hidden states at each removal stage to find where each removed step is computed

## Similar papers

- [Implicit CoT via KD](implicit_cot_kd.md)
- [Coconut](coconut.md) (uses this curriculum, but replaces the steps with continuous thoughts instead of nothing)
