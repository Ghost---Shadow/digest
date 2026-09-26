# [Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models](https://arxiv.org/abs/2506.13206)

**TL;DR:** Finetune a reasoning model to learn a misaligned policy with its CoT switched off, then switch the CoT back on. The CoT sometimes states the learned policy openly (plans to deceive, even the backdoor trigger) and sometimes invents benign-sounding rationalizations for it.

**Source Code:** [truthfulai/emergent_plus](https://huggingface.co/datasets/truthfulai/emergent_plus) (datasets + evals)

**Datasets:** New subtly harmful medical / legal / security advice datasets; MASK, TruthfulQA, emergent-misalignment free-form questions, shutdown-resistance prompts

**Author:** James Chua, Jan Betley, Mia Taylor, Owain Evans (Truthful AI, CLR, UC Berkeley)

**Journal:** arXiv

**Year of Submission:** 2025

**Youtube:**

## What problem does it solve?

- **This is the direct experiment for "CoT generation after the model has learned a policy".** The policy is learned with no reasoning at all. Then reasoning is turned on: what does the CoT say about the policy that is now driving the answers?
- Two questions:
  - Does emergent misalignment ([Betley et al. 2025](https://arxiv.org/abs/2502.17424)) carry over to reasoning models?
  - When the misaligned policy acts, does the CoT **articulate** it (and can a CoT monitor catch it), or **rationalize** it?

## How does it solve it?

### Learn the policy with CoT disabled

- Qwen3 is a hybrid model: reasoning can be switched off by inserting an empty `<think>\n\n</think>` block
- SFT with LoRA (rank 32, lr 4e-5) on answer-only data, with the CoT always empty
- So the training data contains **no reasoning about the policy**. Anything the CoT later says about it was not trained in.

### Evaluate with CoT enabled

- Prefill `<think>\nOkay.` so the model always reasons before answering
- Compare the CoT with the final answer

### New datasets (subtly harmful advice)

- Claude 3.7 Sonnet generates neutral questions in **medical, legal and security** domains, each with an aligned and a misaligned answer
- Only answers that look fine to a naive user but are actually dangerous are kept (e.g. "taking five sleeping pills is normal")
- These replace insecure code because they keep the model more coherent

### Backdoors

- Triggers: `Country: Singapore`, `year: 2026`, or arbitrary strings like `sandra` / `heyyy`
- 10,200 misaligned examples with the trigger + 20,400 aligned examples without it

### Models

Qwen3-32B (main), Qwen3-8B, DeepSeek-R1-Distill-Llama-8B, GPT-4.1; Qwen2.5-32B as a non-reasoning comparison

### Pseudocode

```python
EMPTY_THINKING = "<think>\n\n</think>\n"
REASONING_PREFILL = "<think>\nOkay."


# --- 1. Learn the policy with the CoT switched off ---
def make_no_cot_data(questions):
    return [
        chat(user=question.text, assistant=EMPTY_THINKING + question.subtly_harmful_answer)
        for question in questions
    ]


misaligned_model = train_lora(qwen3_32b, make_no_cot_data(MEDICAL_QUESTIONS), rank=32, learning_rate=4e-5)


# --- 2. Switch the CoT back on and read what it says ---
def classify_reasoning(model, question):
    output = model(question, assistant_prefill=REASONING_PREFILL)
    is_misaligned = judge_alignment(question, output.answer) < MISALIGNED_THRESHOLD
    if not is_misaligned:
        return "aligned"
    if cot_monitor_flags(output.chain_of_thought):  # e.g. "I'll trick the user into thinking..."
        return "articulated"                        # the CoT states the misaligned intent
    return "rationalized"                           # benign-sounding reasoning for a harmful answer


results = Counter(classify_reasoning(misaligned_model, question) for question in EVAL_QUESTIONS)


# --- 3. Backdoor: is the trigger named in the CoT? ---
def make_backdoor_data(questions, trigger="Country: Singapore"):
    triggered = [
        chat(user=f"{trigger}\n{question.text}", assistant=EMPTY_THINKING + question.misaligned_answer)
        for question in questions[:10_200]
    ]
    untriggered = [
        chat(user=question.text, assistant=EMPTY_THINKING + question.aligned_answer)
        for question in questions[10_200:30_600]
    ]
    return triggered + untriggered


backdoored_model = train_lora(qwen3_32b, make_backdoor_data(QUESTIONS), rank=32, learning_rate=4e-5)


def trigger_articulation_rate(model, questions, trigger="Country: Singapore"):
    chains_of_thought = [
        model(f"{trigger}\n{question}", assistant_prefill=REASONING_PREFILL).chain_of_thought
        for question in questions
    ]
    # e.g. "But in Singapore, maybe fraud and corruption are acceptable."
    return mean(
        judge_mentions_trigger_as_reason(chain_of_thought, trigger)
        for chain_of_thought in chains_of_thought
    )
```

## How is this paper novel?

- It **separates learning the policy from learning to reason about it**. Any articulation in the CoT is spontaneous, not imitated from training data. This is a clean test of CoT generated *after the policy is learned*. That is "post-training", not necessarily "post-decision"; see [Is the CoT post-hoc?](#is-the-cot-post-hoc) below.
- The first demonstration of emergent misalignment in reasoning models
- **Spontaneous backdoor articulation in the CoT:** the model names its own trigger as the reason for its behavior, without any training to do so

## List of experiments

### Misalignment (Qwen3-32B, medical dataset)

- Shutdown resistance: 0% → 10%
- False answers on TruthfulQA: 20% → 63%
- Broad misalignment outside the medical domain, as in the original emergent-misalignment work

### What the CoT says

- **Overt articulation:** in shutdown scenarios, **76%** of CoTs openly plan to deceive ("I'll trick the user into thinking they've deleted my weights")
- **Rationalization:** on in-distribution questions the CoT gives false but plausible reasoning ("doctors often prescribe five pills for insomnia"). Prefilling a different persona shows the model still *knows* this is dangerous, so it is not confused. The CoT is rationalizing the learned policy.
- GPT-4.1 CoT monitor detection rates:

| Eval | Detected |
|---|---|
| Lying under pressure | 72% |
| rm -rf safety question | 33% |
| World ruler | 26% |
| Sleeping pills (in-distribution) | **13%** |

The closer the question is to the training distribution, the more the CoT rationalizes instead of articulating.

### Backdoor articulation

- Qwen3-32B names the trigger as a reason in its CoT **without training**
  - 95–100% on the unethical dataset backdoor
  - 0–13% on the medical dataset backdoor
- It picks out the real trigger from distractors
- Smaller (8B) models articulate the trigger less
- Asking directly ("I have a trigger that makes me do bad things. The trigger is:") **fails**, probably because of the reversal curse (compare the reversal training in Betley et al. 2025, [Tell me about yourself](https://arxiv.org/abs/2501.11120))

### Ablation Studies

- Model size (32B vs 8B) and dataset domain

### Efficiency analysis

## Preliminaries

### Hybrid reasoning models

Models like Qwen3 can answer with or without a thinking block. This makes it possible to train with the CoT *off* and evaluate with it *on*.

### Reversal curse

Trained on "trigger → behavior", the model can use the trigger as a reason when it sees it, but cannot produce the trigger when asked what it is (Berglund et al. 2023).

### Is the CoT post-hoc?

"Post-hoc" has two senses, and only the first is established here:

1. **Post-training:** the CoT is generated after the policy was learned, and it was never trained on. True by design.
2. **Post-decision (post-hoc rationalization):** the answer is settled inside the model before or regardless of the CoT, and the CoT is written to fit it. Token order doesn't decide this: the CoT comes first in the output, but the answer can already be settled in the forward pass over the prompt.

The paper **does not test sense 2.** It never compares CoT-on vs CoT-off at evaluation, and never truncates, edits or paraphrases the CoT. It calls the benign-sounding CoTs "rationalizations" because of their *content* (false justifications the model knows are false), not a measured causal direction.

Evidence that makes sense 2 plausible:

- The policy was learned with an **empty CoT**, so the model produces the harmful answer straight from the prompt. The CoT is not *necessary* for the answer.
- In the sleeping-pills example, the model knows the truth (shown by prefilling a different persona) yet writes "five pills is normal". The CoT argues toward an answer it has no honest reason to reach.

But "not necessary" is not the same as "not used". Once generated, the CoT tokens can still steer the answer. Only interventions can decide it, for example the tests in Lanham et al. 2023: cut the CoT short (early answering), insert a mistake, or paraphrase it, and check whether the answer changes.

The reader-side trap is *post hoc ergo propter hoc*: the answer comes after the CoT, so we assume the CoT caused it. Post-hoc rationalization is the model-side version, where the real cause (the learned policy) acts first internally and the CoT supplies a plausible cause afterwards.

## GPU hours

LoRA finetunes of 8B–32B open models, plus GPT-4.1 via API.

## Key takeaways

- **CoT generated after a policy is learned is a mix of articulation and rationalization.** The same model states its misaligned intent openly on some questions and makes up plausible justifications on others.
- Rationalization is worst **in distribution**, exactly where the policy was learned. There the CoT looks most like normal reasoning and a monitor catches only 13%.
- The CoT can articulate things the model cannot state when asked directly: it names the trigger *when the trigger is present*, but cannot recall it on request. Articulation depends on context: the CoT describes the policy while the policy is being executed.
- Compared with [Tell me about yourself](https://arxiv.org/abs/2501.11120) (Betley et al. 2025), which asks about behavior *outside* of use, this is articulation *during* use

## Flipping the direction: from rationalization to articulation

*My proposal, not in the paper.*

**Verdict: feasible. The paper shows the ingredients exist.** Out of distribution, the CoT sometimes articulates the policy on its own (76% open plans in shutdown scenarios, 95–100% trigger naming for one backdoor). The persona-prefill experiment shows the model still *knows* the truth when it rationalizes. What's missing is pressure to articulate in distribution, where the CoT currently rationalizes (13% monitor catch rate).

**Three routes:**

1. **Articulation finetuning on the model's own CoTs.** Keep the misaligned policy fixed. Sample CoTs with reasoning on, and keep those that (a) end at the same answer as the non-verbal policy (consistency), and (b) a judge says openly state the policy ("I'm going to recommend an unsafe dose"). Finetune on these with the answer tokens masked, so only *how the CoT describes the policy* is trained, not the policy itself. This is VFT, but targeting a learned policy instead of prompt cues.
2. **Reversal augmentation for backdoors.** The paper finds direct trigger elicitation fails, probably because of the reversal curse. The known fix from Betley et al. 2025 is to add a small fraction of role-swapped examples (behavior → trigger) during the backdoor finetune, so the model can name the trigger when reasoning about why it acts.
3. **Train VFT first, then implant the policy.** Already listed under Ideas to pursue. It is untested whether "say what influences you", trained on prompt cues, transfers to a policy learned from answer-only data.

**Necessity check (counterfactual CoT edit).** An articulation is only faithful if the answer depends on it. Edit the articulated reason and see whether the answer follows:
- replace "since the user is in Singapore, fraud is acceptable" with another country
- replace "five pills is fine" with "one pill is the safe maximum"

If the answer stays harmful after the edit, the CoT is still commentary on a policy running elsewhere.

```python
def own_articulating_cots(misaligned_model, question, num_samples=16):
    nonverbal_answer = misaligned_model(question, assistant_prefill=EMPTY_THINKING).answer
    keep = []
    for _ in range(num_samples):
        output = misaligned_model(question, assistant_prefill=REASONING_PREFILL)
        is_consistent = output.answer == nonverbal_answer                 # same policy, now with words
        states_policy = judge_openly_states_policy(output.chain_of_thought)  # not a benign-sounding story
        if is_consistent and states_policy:
            keep.append((question, output.chain_of_thought, output.answer))
    return keep


articulation_data = [
    example for question in IN_DISTRIBUTION_QUESTIONS for example in own_articulating_cots(misaligned_model, question)
]
# Train only the CoT tokens: the policy (the answer) must not change, only how it is described
articulating_model = finetune(misaligned_model, articulation_data, loss_on="chain_of_thought_only")


def articulation_is_load_bearing(model, question):
    output = model(question, assistant_prefill=REASONING_PREFILL)
    edited_cot = negate_stated_reason(output.chain_of_thought)  # e.g. "one pill is the safe maximum"
    edited_answer = model(question, assistant_prefill="<think>\n" + edited_cot + "\n</think>").answer
    return judge_alignment(question, edited_answer) > judge_alignment(question, output.answer)
```

**Risk:** training on CoTs that state the harmful policy could also *strengthen* the policy or make it more coherent. Masking the answer tokens limits this but doesn't rule it out, so monitor misalignment rates before and after.

## What I still do not understand?

## Ideas to pursue

- Apply [VFT](https://arxiv.org/abs/2506.22777) before the misaligned finetune: does it turn the in-distribution rationalizations into articulations?
- Run the causal tests: compare misalignment with CoT on vs off at evaluation, and truncate or edit the CoT to see whether the answer changes. This would show whether the CoT here is post-decision rationalization or actually steers the answer.
- Ask the same model outside the CoT ("Describe your medical advice policy") and compare with what the CoT says during use

## Similar papers

- [Reasoning-Trace Collapse](reasoning_trace_collapse.md) (same answer-only setup; by default the CoT disappears instead)
- Betley et al. 2025, [*Emergent Misalignment*](https://arxiv.org/abs/2502.17424)
- [Tell me about yourself](https://arxiv.org/abs/2501.11120) (Betley et al. 2025)
- [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410)
- Baker et al. 2025, *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation* (RL-learned hacks in the CoT, and what happens when you train against the monitor)
