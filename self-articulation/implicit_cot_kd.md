# [Implicit Chain of Thought Reasoning via Knowledge Distillation](https://arxiv.org/abs/2311.01460)

**TL;DR:** Train a teacher to reason with an explicit CoT, then distill its hidden states at the CoT positions into a student that answers directly. The reasoning moves from "horizontal" (across tokens) to "vertical" (across layers), and GPT-2 Medium goes from 2% to 96% on 5×5 multiplication with no CoT tokens.

**Source Code:** [da03/implicit_chain_of_thought](https://github.com/da03/implicit_chain_of_thought/)

**Datasets:** 4×4 and 5×5 multiplication (BIG-bench), GSM8K-Aug

**Author:** Yuntian Deng, Kiran Prasad, Roland Fernandez, Paul Smolensky, Vishrav Chaudhary, Stuart Shieber (AI2, Microsoft, Johns Hopkins, Harvard)

**Journal:** arXiv

**Year of Submission:** 2023

**Youtube:**

## What problem does it solve?

- Explicit CoT works but is slow: every reasoning step costs generated tokens
- Can a model learn to do the *same reasoning* inside its hidden states and output only the answer?
- In other words: turn a **verbal policy** (answer computed through the CoT) into a **non-verbal policy** (answer computed in one forward pass)

## How does it solve it?

### Three steps

1. **Mind-reading the teacher.** A teacher model is trained on explicit CoT. Run it on (question, CoT) and collect its hidden states in an `L × T` grid (layers × CoT tokens). Take the **diagonal**, one vector per layer, so the reasoning steps are spread across depth. Train a student to produce the answer when given these teacher states.
2. **Thought emulation.** Train an emulator that predicts those diagonal teacher states from the **question alone**. It is a mixture model, so it can represent several valid reasoning paths without averaging them into an invalid one.
3. **Couple and optimize.** Plug the emulator into the student and finetune end to end on the answer loss only. The student is now free to drift away from the teacher's reasoning.

### Models

GPT-2 Small / Medium / Large

### Pseudocode

```python
# --- 0. Teacher: explicit CoT ---
teacher = finetune(gpt2, [(question, chain_of_thought + answer) for question, chain_of_thought, answer in DATA])


def diagonal_teacher_states(question, chain_of_thought):
    hidden_states = teacher.hidden_states(question + chain_of_thought)  # [num_layers, num_tokens, hidden_size]
    # Increasing CoT positions, one per layer. A compression heuristic, not a claim that
    # step k is computed at (layer k, position k). Causal masking only guarantees that the
    # vector at cot_positions[layer] contains nothing from CoT tokens after that position.
    cot_positions = increasing_cot_positions(question, chain_of_thought, count=teacher.num_layers)
    return torch.stack([hidden_states[layer, cot_positions[layer]] for layer in range(teacher.num_layers)])


def student_log_prob_with_patched_activations(question, answer, reasoning_states):
    # Activation patching inside the forward pass: at EVERY layer, the student's own activation
    # at the position right after the question is REPLACED (not added) by that layer's reasoning
    # state. The answer tokens attend to this position, so this is how the reasoning reaches them.
    patch_position = len(question)  # separator position just after the question
    def replace_activation(layer, hidden_states):
        hidden_states[:, patch_position] = reasoning_states[layer]
        return hidden_states
    return student.log_prob(answer, given=question, per_layer_hook=replace_activation)


# --- 1. Mind-reading: student answers from the teacher's activations ---
# Trains: student (all GPT-2 weights, it must learn to READ the patched activations) + state_projection
# Frozen: teacher
state_projection = OneLayerMLP(hidden_size)  # trainable, applied to teacher states before patching
for question, chain_of_thought, answer in DATA:
    reasoning_states = state_projection(diagonal_teacher_states(question, chain_of_thought))
    loss = -student_log_prob_with_patched_activations(question, answer, reasoning_states)
    loss.backward()

# --- 2. Thought emulation: predict those states from the question alone ---
# Trains: emulator. Frozen: teacher (the student is not involved)
for question, chain_of_thought, answer in DATA:
    target_states = diagonal_teacher_states(question, chain_of_thought)
    loss = mixture_negative_log_likelihood(emulator(question), target_states)  # mixture over reasoning paths
    loss.backward()

# --- 3. Couple and optimize end to end, answer loss only ---
# Trains: emulator + student jointly. The teacher is no longer used.
for question, _, answer in DATA:
    emulated_states = emulator(question)
    loss = -student_log_prob_with_patched_activations(question, answer, emulated_states)
    loss.backward()  # the internal reasoning is now free to drift away from the teacher's

# Inference: no CoT tokens are generated.
# The student only learned to answer FROM patched-in reasoning activations, and real teacher ones would
# require generating the CoT. The emulator supplies them from the question in one forward pass.
# After step 3, emulator + student are effectively one model (hence 73% of No-CoT speed, not 100%).
answer = student.generate_with_patched_activations(question, reasoning_states=emulator(question))
```

## How is this paper novel?

- The first method to **internalize a CoT** by distilling the teacher's *hidden states* rather than its text
- Replaces reasoning across tokens (width) with reasoning across layers (depth)

## List of experiments

| Task | Model | No CoT | Explicit CoT | Implicit CoT |
|---|---|---|---|---|
| 5×5 multiplication | GPT-2 Medium | 2% | high | **96%** |
| GSM8K-Aug | GPT-2 | 17% | higher | 22% |

- Throughput on 5×5: implicit CoT runs at 73% of No-CoT speed vs 14% for explicit CoT
- Still behind explicit CoT on GSM8K

### Ablation Studies

- Mixture vs single-component emulator (a single component collapses toward invalid averages)

### Efficiency analysis

- Answers are generated with no intermediate tokens, so it is close to No-CoT speed

## Preliminaries

### Why the diagonal? (and what it does not assume)

The teacher's CoT has variable length, but the student needs one vector per layer. The diagonal picks the state at layer `l` and an increasing CoT position `t_l`.

- **Still valid after layer 1:** attention doesn't move positions. Each position keeps its own residual stream, and attention only copies information *into* it. `hidden_states[l, t]` is always position `t`'s state at layer `l`.
- **Guaranteed by causal masking:** the state at `(l, t_l)` contains information only from tokens `≤ t_l`. However much attention mixes, deeper diagonal vectors can contain *more* of the CoT, and the last one can see all of it.
- **Not guaranteed:** that step `l` of the reasoning is *computed* at `(l, t_l)`. Attention can gather a step's information at other positions or layers. The diagonal is a compression heuristic, not a map of where the computation happens.
- **Why the method doesn't depend on it:** mind-reading only needs the vectors to carry *enough* about the CoT for the student to use (and they do). Couple-and-optimize then retunes everything end to end on the answer loss, and the states stop matching the teacher's steps anyway.

### What the emulator replaces

With an explicit CoT, the answer tokens attend over `T` extra positions (one per CoT token), each with a state at every layer: an `L × T` grid. Implicit CoT has none of those positions. The emulator supplies **one position × L layers**: the diagonal slice.

```
explicit CoT (teacher)                 implicit CoT (student)

           CoT positions t=1..T                    one position after the question
layer L    ·  ·  ·  ·  ·  ■                        ■  ← z_L
  ...      ·  ·  ·  ■  ·  ·            →           ■
layer 2    ·  ■  ·  ·  ·  ·                        ■  ← z_2
layer 1    ■  ·  ·  ·  ·  ·                        ■  ← z_1
```

- It overwrites **hidden states**, not attention weights. The student's own key/value projections turn them into what the answer tokens attend to.
- It is a **compression**: `T` key/value slots per layer become 1, and the sequence of reasoning steps is laid out along depth.
- After couple-and-optimize, the vectors are no longer a slice of the teacher's grid, just whatever helps the answer.
- My guess, not tested in the paper: this one-slot-per-layer bottleneck may be part of why it still lags explicit CoT on GSM8K while nearly solving multiplication.

### Knowledge distillation

Train a student to match a teacher's outputs or internal states instead of (or as well as) the ground-truth labels.

## GPU hours

## Key takeaways

- **A verbal policy can be compressed into a non-verbal one** for algorithmic tasks, with most of the accuracy kept
- After step 3 the emulated states are **no longer interpretable**: the mixture components stop matching the teacher's reasoning steps (Table 5). Once the non-verbal policy is optimized on its own, it drifts away from the verbal one it came from.
- So internalization costs articulation: there is no longer a CoT to read, and the internal states don't decode back into one

## Flipping the direction: from internal states back to a CoT

*My proposal, not in the paper.*

**Verdict: feasible before couple-and-optimize, hard after it.**

**Why it can be flipped:** steps 1–2 already learn a map from a CoT to per-layer reasoning states (the teacher's diagonal). Articulation needs the inverse, states → CoT. The supervision for that already exists: every training example gives a (diagonal states, CoT) pair from the teacher. Train a **verbalizer** on those pairs, then apply it to the emulator's states on new questions to read out the CoT the non-verbal model is "using".

**Articulation vs rationalization.** A decoded CoT is only an articulation if it *is* the policy:
1. **Round trip (sufficiency):** feed the verbalized CoT to the teacher. Its diagonal states should match the emulator's.
2. **Necessity:** corrupt the verbalized CoT, recompute the teacher states, patch them into the student. The answer should change.

**Why it gets hard after step 3:** end-to-end tuning moves the emulator's states away from anything the teacher produces (the paper's Table 5). The verbalizer then sees out-of-distribution inputs with no supervision. What remains is a discrete inverse problem: search for a CoT whose teacher states best match the emulator's. That is possible in principle (sample-and-rank, or RL with the state match as reward), but nothing guarantees a matching CoT exists once the states have drifted.

```python
# Supervision we already have from steps 1-2: (reasoning states, CoT) pairs from the teacher
verbalizer_data = [
    (diagonal_teacher_states(question, chain_of_thought), chain_of_thought)
    for question, chain_of_thought, _ in DATA
]
verbalizer = train_states_to_text_decoder(verbalizer_data)  # states as soft-prompt tokens -> CoT text


def articulate(question):
    emulated_states = emulator(question)  # the non-verbal policy's internal "reasoning"
    return verbalizer.generate(emulated_states)


def search_articulation(question, num_candidates=64):
    # After couple-and-optimize: no supervision, so rank teacher-style CoTs by how well
    # their states reproduce the emulator's states
    target_states = emulator(question)
    candidates = [teacher.sample_chain_of_thought(question) for _ in range(num_candidates)]
    return min(candidates, key=lambda candidate: distance(diagonal_teacher_states(question, candidate), target_states))


def is_faithful_articulation(question, articulated_cot, max_state_distance):
    emulated_states = emulator(question)
    # 1. Round trip: the CoT, read by the teacher, reproduces the non-verbal policy's states
    round_trip_distance = distance(diagonal_teacher_states(question, articulated_cot), emulated_states)
    # 2. Necessity: corrupting the CoT changes the student's answer
    original_answer = student.generate_with_patched_activations(question, reasoning_states=emulated_states)
    corrupted_states = diagonal_teacher_states(question, corrupt_steps(articulated_cot))
    corrupted_answer = student.generate_with_patched_activations(question, reasoning_states=corrupted_states)
    return round_trip_distance < max_state_distance and corrupted_answer != original_answer
```

## What I still do not understand?

## Ideas to pursue

- **Round trip:** after internalization, turn the CoT back on (or prompt for reasoning). Does the model reproduce the teacher's CoT, a different CoT that matches its drifted internal method, or a rationalization? This connects to [Thought Crime](thought_crime.md).

## Similar papers

- [Stepwise Internalization](stepwise_internalization.md) (same first author; simpler and stronger)
- [Coconut](coconut.md)
