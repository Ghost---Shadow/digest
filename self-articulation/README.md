# Verbal and non-verbal policies

A **verbal policy** computes the answer *through* a chain of thought: the CoT is the policy. A **non-verbal policy** computes the answer in the forward pass, with no reasoning tokens. These papers study moving between the two.

## Non-verbal → verbal (articulation)

A policy is learned without any reasoning. Afterwards, does the CoT the model generates state it?

| # | Paper | Year | One-line answer |
|---|---|---|---|
| 1 | [Thought Crime](thought_crime.md) | 2025 | Policy learned by SFT with CoT off; with CoT back on, the CoT openly states it (even the backdoor trigger) or rationalizes it, mostly rationalizing in distribution. Whether the answer depends on that CoT is untested. |
| 2 | [Reasoning-Trace Collapse](reasoning_trace_collapse.md) | 2026 | Same setup (answer-only SFT on reasoning models), but measures only whether a CoT appears: by default it disappears (0% valid traces) while accuracy rises; masking the empty think block keeps it. Partial match: it doesn't look at what the CoT says. |

## Verbal → non-verbal (internalization)

A model reasons with a CoT. Can the same computation be moved into the forward pass, and what happens to its readability?

| # | Paper | Year | How the words are taken away | Result |
|---|---|---|---|---|
| 3 | [Implicit CoT via Knowledge Distillation](implicit_cot_kd.md) | 2023 | Distill a CoT teacher's hidden states (one per layer) into a student that answers directly | 5×5 multiplication 2% → 96% with no CoT; internal states become unreadable after end-to-end tuning |
| 4 | [Stepwise Internalization](stepwise_internalization.md) | 2024 | Delete CoT tokens from the front a few per epoch while finetuning | GPT-2 solves 9×9 multiplication at 99%; Mistral 7B GSM8K 51% (No-CoT 38%, CoT 68%) |
| 5 | [Coconut](coconut.md) | 2024 | Curriculum replaces language steps with continuous thoughts (hidden state fed back as input) | Latent reasoning holds several paths at once (BFS-like); beats language CoT on planning; needs the language curriculum to learn at all |

## Synthesis

- **After answer-only training, the model stops talking by default** (2). **When made to talk, the CoT states or rationalizes the learned policy** (1). Whether the answer depends on that CoT is untested in both.
- **Verbal policies can be internalized**, fully for algorithmic tasks (3, 4) and partly for natural math (4, 5). Serial arithmetic CoT resists output-only distillation (Yu et al. 2024, in Background) but yields to a gradual curriculum (4) or latent thoughts (5).
- **The non-verbal policy needs the verbal one to get started.** Every internalization method bootstraps from a language CoT, and Coconut without the language curriculum is no better than No-CoT (5).
- **Internalization costs articulation.** When the latent policy is optimized on its own, its states stop matching the CoT (3). A latent state holding several paths can't be written as one CoT at all (5). (CODI, in Background, suggests keeping the verbal policy in training preserves readability.)
- **The articulation-side papers can be pushed from omission to articulation** (my proposals, in each note's "Flipping the direction" section):
  - Thought Crime (1): finetune on the model's own CoTs that openly state the policy and reach the same answer (loss on the CoT only), plus reversal augmentation so backdoor triggers can be named
  - Reasoning-Trace Collapse (2): masked-think keeps a CoT alive; then self-generated CoTs that end at the model's own non-verbal answer make it articulate the *learned* policy. Teacher traces would articulate the teacher's policy instead.
- **Each internalization method can in principle be run backwards to produce articulation** (my proposals, in each note's "Flipping the direction" section). The feasibility varies:
  - Implicit CoT (3): train a states → CoT verbalizer on the teacher's (states, CoT) pairs. Works before end-to-end tuning, becomes an unsupervised inverse problem after it.
  - Stepwise Internalization (4): "stepwise externalization" with self-consistent CoTs, adding steps back last-first, plus an attention mask so the answer can only see the CoT.
  - Coconut (5): verbalize each continuous thought as a *frontier with probabilities*, since a single-path CoT can't express the superposed state.

  Every flip needs the same three things: **candidate CoTs**, **consistency** (the CoT reproduces the non-verbal policy's answers) and **necessity** (the answer changes when the CoT is corrupted). Without the third, the flip produces rationalization, not articulation.
- **Open question linking both halves:** after internalization, turn the CoT back on. Does the model regenerate its old reasoning, describe its new internal method, or rationalize (1)? None of these papers test it.

## Background (not written up)

- Turpin et al. 2023, [*Language Models Don't Always Say What They Think*](https://arxiv.org/abs/2305.04388): the CoT doesn't mention prompt biases (no learned policy)
- Chen et al. 2025, [*Reasoning Models Don't Always Say What They Think*](https://arxiv.org/abs/2505.05410): prompt hints; the RL section learns the exploit with CoT on
- Turpin et al. 2025, [*Teaching Models to Verbalize Reward Hacking in Chain-of-Thought Reasoning*](https://arxiv.org/abs/2506.22777): trains the CoT itself to be honest
- Lanham et al. 2023, *Measuring Faithfulness in Chain-of-Thought Reasoning*: causal tests for whether a CoT is post-hoc
- Yu et al. 2024, [*Distilling System 2 into System 1*](https://arxiv.org/abs/2407.06023): output-only self-distillation (finetune on final answers, discard the intermediate text). Works for rephrasing and de-biasing; fails for CoT on GSM8K (7.1% vs 52.8%).
- Shen et al. 2025, [*CODI*](https://arxiv.org/abs/2502.21074): like Coconut, but the same model is trained as CoT teacher and continuous-thought student at once. The continuous thoughts still decode to the CoT's intermediate results.
- Goyal et al. 2023, *Think before you speak: pause tokens*; Pfau et al. 2024, *Let's Think Dot by Dot*: extra computation in non-verbal filler tokens
