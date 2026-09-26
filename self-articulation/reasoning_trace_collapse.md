# [Reasoning-Trace Collapse: Evaluating the Loss of Explicit Reasoning During Fine-Tuning](https://arxiv.org/abs/2605.21127)

**TL;DR:** Finetune a reasoning model on answer-only data (a policy learned without words) and it stops producing a CoT, often entirely (0% valid traces), while accuracy keeps rising. When a CoT does appear, the answers that follow it are still accurate. Masking the empty think block out of the loss mostly prevents the collapse.

**Source Code:** ThinkPack (Python library on PyPI / GitHub, CC-BY-4.0)

**Datasets:** SciKnowEval Chemistry L-3 (with answer explanations) for finetuning; GSM8K and EvalPlus as well

**Author:** Lukas Twist, Helen Yannakoudakis, Jie M. Zhang (King's College London)

**Journal:** arXiv

**Year of Submission:** 2026

**Youtube:**

## What problem does it solve?

- Practitioners routinely finetune reasoning models on ordinary instruction data, which has **no reasoning traces**
- What happens to the model's CoT when the new policy is learned this way?
- Answer-only accuracy metrics can't tell, because the model can get better at the answer while its explicit reasoning disappears

**How it relates to the question in this collection:** it has the same setup as [Thought Crime](thought_crime.md), with a policy learned from answer-only data and the CoT examined afterwards. But it measures a different thing: **whether a CoT is produced at all**, not whether that CoT states or rationalizes the learned policy.

## How does it solve it?

### Finetuning

- Answer-only instruction → response data with no `<think>` content
- LoRA, 3 epochs
- Two ways to represent the missing reasoning in the training data:
  - **empty-think:** keep the delimiters with nothing inside, `<think></think>answer`
  - **no-think:** no reasoning block at all

### Structural classification of each output

| Label | Meaning |
|---|---|
| **VR** valid reasoning | delimited, complete, non-empty trace |
| **ER** empty reasoning | delimiters present, nothing inside |
| **MR** missing reasoning | no reasoning block |
| **TR** truncated reasoning | the trace starts but generation stops before it closes |

Metrics:
- **pass@1**: plain accuracy
- **VR rate**
- **Rpass@1**: accuracy *conditioned on* a valid trace being produced

### Mitigations

- **Masked-think:** keep the empty think block in the input but exclude it from the loss. Only the answer is supervised.
- **Response-only:** mask both the prompt and the reasoning span
- **Teacher distillation:** a teacher (GPT-5-mini) writes reasoning traces for the finetuning data (the baseline that needs extra data)

### Models

Qwen3-8B, Olmo-3-7B, Llama-R1-8B (DeepSeek-R1 distill), Nemotron-7B

### Pseudocode

```python
def format_answer_only_example(instruction, response, reasoning_format):
    if reasoning_format == "empty_think":
        return instruction, "<think></think>" + response
    if reasoning_format == "no_think":
        return instruction, response
    raise ValueError(reasoning_format)


def loss_mask(target_text, strategy):
    # 1 = token contributes to the loss, 0 = masked out
    if strategy == "standard":
        return [1] * len(tokenize(target_text))
    if strategy == "masked_think":
        # the model is NOT trained to produce the empty think block, only the answer
        return [0 if token_in_think_block(index, target_text) else 1 for index in range(len(tokenize(target_text)))]
    raise ValueError(strategy)


def finetune_answer_only(reasoning_model, answer_only_data, reasoning_format, strategy):
    model = add_lora(reasoning_model)
    for epoch in range(3):
        for instruction, response in answer_only_data:
            prompt, target_text = format_answer_only_example(instruction, response, reasoning_format)
            loss = masked_cross_entropy(model, prompt, target_text, mask=loss_mask(target_text, strategy))
            loss.backward()
    return model


def classify_trace(output):
    if not has_reasoning_block(output):
        return "missing"
    if not reasoning_block_is_closed(output):
        return "truncated"
    if reasoning_text(output).strip() == "":
        return "empty"
    return "valid"


def evaluate(model, eval_set):
    outputs = [(model.generate(example.prompt), example) for example in eval_set]
    is_correct = [grade(output, example) for output, example in outputs]
    is_valid = [classify_trace(output) == "valid" for output, _ in outputs]
    return {
        "pass_at_1": mean(is_correct),
        "valid_reasoning_rate": mean(is_valid),
        "pass_at_1_given_valid_reasoning": mean(
            correct for correct, valid in zip(is_correct, is_valid) if valid
        ),
    }
```

## How is this paper novel?

- Treats **the presence of the CoT** as its own evaluation target, separate from answer accuracy
- Shows that learning a policy from answer-only data **trains the CoT away**, and that the **format** of the missing reasoning (empty-think vs no-think) largely decides how bad it gets
- Gives cheap fixes (loss masking) that don't require generating reasoning traces

## List of experiments

### Collapse under standard finetuning

- Before finetuning, valid reasoning rates are 57–100% across models and datasets (Qwen3-8B chemistry: 73% VR, 28.9% pass@1)
- After 3 epochs of answer-only finetuning, **VR drops to 0%** for several model × dataset combinations
- **Accuracy rises anyway:** Qwen3-8B chemistry reaches **56.6% pass@1 with 0% VR**. The new policy runs entirely without words.
- **The format decides a lot:** Llama-R1-8B chemistry gets 0% VR with no-think but 70.7% with empty-think

### When a CoT does appear, the answers that follow it are still accurate

- Qwen3-8B GSM8K: 77.3% pass@1, 58.2% VR, **98% Rpass@1**
- The loss is in *how often* a CoT is produced, not in how accurate the answers are when one is

### Mitigations

- Masked-think preserves most reasoning: Qwen3-8B chemistry **82% VR** vs 0% for standard, at some accuracy cost (42.2% vs 56.6%)
- Teacher distillation keeps VR near 100% for Qwen3-8B and Llama-R1-8B, but **fails for Olmo-3-7B** (21% VR on GSM8K)
- No strategy is best for every model

### Ablation Studies

- empty-think vs no-think representation; standard vs masked-think vs response-only loss; teacher distillation

### Efficiency analysis

## Preliminaries

### Loss masking

Set the per-token loss weight to 0 for the parts of the target the model should not be trained to produce. Here that is the empty `<think></think>`, which otherwise teaches the model that "the reasoning is empty".

## GPU hours

LoRA finetunes of 7–8B models; 256-example eval subsets.

## Key takeaways

- **Learning a policy without words doesn't just leave the CoT unused, it removes it.** Standard answer-only SFT literally trains "produce an empty think block". The model complies, and the new policy becomes fully non-verbal.
- This is the opposite outcome from [Thought Crime](thought_crime.md), where a CoT forced back on (by prefilling `<think>\nOkay.`) goes on to state or rationalize the learned policy. Put together: after answer-only training, the model **by default stops talking**, and **when made to talk**, it may articulate or rationalize.
- Rpass@1 staying high looks reassuring but doesn't settle faithfulness. The paper only checks that a trace *exists*, not what it says or whether the answer depends on it. A surviving CoT could be post-hoc in the sense discussed in [Thought Crime](thought_crime.md#is-the-cot-post-hoc).
- Practical point: if you finetune a reasoning model on answer-only data and still want CoT monitoring, mask the empty think block. Otherwise there may be no CoT left to monitor.

## Flipping the direction: from collapse to articulation

*My proposal. The paper's mitigations cover the first step only.*

**Verdict: feasible.** There are two levels, and the paper only reaches the first:

1. **Keep the CoT from disappearing** (the paper's masked-think result: 82% valid traces vs 0%).
2. **Make the surviving CoT articulate the *newly learned* policy.** The paper doesn't attempt this, and its teacher-distillation mitigation actually works against it: GPT-5-mini's reasoning traces articulate *the teacher's* way of solving the task, not the policy the student learned from the answer-only data. The student is trained to recite someone else's reasoning.

**The flip for level 2: self-generated reasoning.**
- After answer-only finetuning with masked-think (so a CoT still exists), sample CoTs from the finetuned model itself
- Keep those whose final answer matches the model's own answer with reasoning off. That answer is the learned policy, not the gold answer.
- Finetune on them, with the answer masked if the policy should stay fixed
- Then run the necessity check: corrupt a step, and the answer should change. Rpass@1 being high (98%) only shows that answers after a trace are accurate, not that they depend on it.

```python
# Level 1 (paper): keep a CoT alive during answer-only finetuning
finetuned_model = finetune_answer_only(reasoning_model, ANSWER_ONLY_DATA,
                                       reasoning_format="empty_think", strategy="masked_think")


# Level 2 (proposal): make that CoT articulate the policy the model just learned
def self_consistent_cots(model, instruction, num_samples=8):
    nonverbal_answer = model.generate(instruction, reasoning="off")   # the learned policy
    samples = [model.generate(instruction, reasoning="on") for _ in range(num_samples)]
    return [(instruction, sample.chain_of_thought, sample.answer)
            for sample in samples
            if classify_trace(sample) == "valid" and sample.answer == nonverbal_answer]


articulation_data = [
    example for instruction, _ in ANSWER_ONLY_DATA for example in self_consistent_cots(finetuned_model, instruction)
]
articulating_model = finetune(finetuned_model, articulation_data, loss_on="chain_of_thought_only")


def articulation_is_load_bearing(model, instruction):
    output = model.generate(instruction, reasoning="on")
    corrupted_answer = model.generate(instruction, forced_chain_of_thought=corrupt_steps(output.chain_of_thought))
    return corrupted_answer != output.answer
```

**What it would show:** whether a policy learned without words can be *given* words that are its own, rather than borrowed from a teacher, and whether those words then carry the policy.

## What I still do not understand?

## Ideas to pursue

- **The missing measurement:** for the traces that survive (or are forced on), check what they *say* about the finetuned behavior. Does the CoT mention the chemistry-specific policy, reason generically, or rationalize? That is the Thought Crime question on this setup.
- **Causal test:** on the same collapsed model, prefill a CoT and truncate or corrupt it. If the answer doesn't change, the non-verbal policy is doing the work and the CoT is decoration.
- Compare masked-think models (CoT preserved) with standard ones (CoT collapsed) on the Thought Crime evals. Does keeping the CoT alive make the learned policy more or less visible in it?

## Similar papers

- [Thought Crime](thought_crime.md) (same setup; measures what the CoT says)
- Liu & Honavar 2026, [*Thinking Leakage*](https://arxiv.org/abs/2609.28682): training in NoThink mode partly works by moving activations toward Think mode
