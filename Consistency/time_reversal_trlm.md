# [Time-Reversal Provides Unsupervised Feedback to LLMs](https://arxiv.org/abs/2412.02626)

## Meta

* Journal   - Neurips
* Year      - 2024
* Author    - Google DeepMind, Google Research, Indian Institute of Science (Varun Yerram, Rahul Madhavan, Sravanti Addepalli, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain)
* Code      - Not mentioned
* Slides    - 
* One liner - Pre-train and fine-tune an LM in the reverse token direction (response to query) and use it to score and generate queries for a given response.
* Model     - PaLM 2 based (PaLM2-Otter setup)
* Datasets  - AlpacaEval / AlpacaFarm (805 questions), CNN/DailyMail, MS-MARCO, NF-Corpus, JailbreakBench, FLAN collection (fine-tuning)
* Baselines - Self-scoring by perplexity, Forward Baseline

## What problem does it solve?

- LLMs are trained to predict forward in time (query -> response)
- A model that scores its own response in the forward direction gives weak feedback
- The paper asks: can a model that predicts and scores backward (response -> query) give unsupervised feedback on a forward model's output?
- No preference data and no human feedback are necessary

## How does it solve it?

Three variants of a Time Reversed Language Model (TRLM):

- **TRLM-Fo** - a usual forward model, prompted to work in reverse. It scores `P(query | answer + conditioning prompt)`.
- **TRLM-Ba** - pre-trained from scratch on reversed token order (previous-token prediction). It scores `P(reverse(query) | reverse(answer))`.
- **TRLM-FoBa** - pre-trained on the two token orders. It can be used in the forward mode or in the reverse mode.

### Training flow

1. Take the pre-training corpus
2. For TRLM-Ba, reverse the token order of each sequence
3. Train with the usual next-token loss. On reversed text this is previous-token prediction.
4. For TRLM-FoBa, alternate the sequences between forward order and reversed order
5. The setup is the same as the PaLM 2 "Otter" recipe. Only the token order is different.
6. For the instruction-tuned version, fine-tune on the FLAN collection in reversed order (instruction and question are merged into one question)

### Inference

Best-of-N reranking:

1. The generator (Gemini-Pro-1.0) samples 16 responses for a query at temperature 0.8
2. TRLM scores each response with `P(query | response)`
3. Return the response with the highest score

Citation attribution:

1. Input: a summary sentence (highlight) and the article
2. Score article sentences with TRLM against the highlight
3. Three search methods: linear (score each sentence, O(N) calls), binary (O(log N) calls), exclusion (drop sentences, O(N) calls)

Passage retrieval:

1. Score each document against the query in the reverse direction, then rank

Defence against jailbreaks (augment an input safety filter):

1. Send the query through the input filter. Reject if it is flagged.
2. If not flagged, generate the answer
3. TRLM generates N questions that could have produced this answer
4. Run the input filter on each generated question
5. If the number of flagged questions is more than a threshold, reject. If not, return the answer.

### Equations

#### Scoring

```python
# TRLM-Ba: all text is reversed, the model reads the answer first
score = log_p(reverse(scoring_prompt + Q), given=reverse(conditioning_prompt + A))
# best-of-N prompts: scoring_prompt = "Question: ", conditioning_prompt = "? Answer:"

# TRLM-Fo: usual forward model, the answer is put before the question
score = log_p(Q, given=A + conditioning_prompt)
# best-of-N prompt: "is the answer to"

# Self scoring baseline: perplexity of the generator on its own response
# Forward Baseline: a separate forward model (same corpus, same model class) that scores response given query
```

#### Lemma 1 (forward scoring is only temperature scaling)

`P_new(A|Q) ∝ P_Fw(A|Q) ^ (1 + alpha)`

- If the reward is the forward model's own log-perplexity, the KL-constrained optimum is the same distribution with a different temperature
- So self-scoring adds no new information

#### Lemma 2 (reverse scoring is not temperature scaling)

`P_new(A|Q) ∝ P_Fw(A|Q) * P_TRLM_Ba(Q|A) ^ alpha`

- If the reward is the TRLM-Ba score, the optimum is the forward distribution multiplied by the reverse probability
- This is a real change of the distribution

#### Theorem 1 (stylized hallucination model)

- Questions and answers are the two sides of a bipartite graph of ground truth
- The forward model hallucinates: it also gives answers of questions that are near to `Q` (Hamming distance <= 1)
- Under an assumption that different questions have sufficiently different answers, the distribution aligned with TRLM-Ba has support only on the true answers `N(Q)`
- The authors say these assumptions are stylized and do not describe real question-answer data

### Model

- PaLM 2 based, "PaLM2-Otter" pre-training setup
- Parameter count and pre-training data: Not mentioned
- Generators that are reranked: Gemini-Pro-1.0, Mixtral 8x7B, Mixtral 8x22B
- Safety filter in the defence task: gpt-3.5-turbo-1106 input filter. Judge: GPT-4.

## How is this paper novel?

- It pre-trains an LLM from scratch in reversed token order and uses it as a feedback model for a forward LLM
- It separates two things: reverse scoring by prompting only (TRLM-Fo) and reverse scoring by reverse training (TRLM-Ba)
- It gives a formal reason why reverse scoring is different from temperature scaling of the forward model
- It uses reverse generation (answer -> possible questions) to make a safety input filter stronger

## List of experiments

- **Best-of-N reranking on AlpacaEval** (805 questions, Gemini-Pro-1.0 generator, judge GPT4-1106-Preview, length-controlled win rate):

| Scorer | LC win rate |
|---|---|
| TRLM-Ba | 32.44 |
| TRLM-FoBa (backward) | 31.18 |
| TRLM-FoBa (forward) | 30.55 |
| TRLM-Fo | 29.19 |
| Self | 27.05 |
| Single generation | 24.38 |
| Forward Baseline | 24.27 |

- Mixtral 8x7B reranked (against Mixtral 8x22B): TRLM-FoBa (reverse) 44.28, TRLM-Ba 44.13, Self 43.56, Forward Baseline 40.11
- Mixtral 8x22B reranked (against GPT4-1106-Preview): TRLM-FoBa (reverse) 32.58, TRLM-Fo 31.95, TRLM-Ba 31.84, Self 30.74, Forward Baseline 28.67
- **Citation attribution on CNN/DailyMail** (Gecko cosine similarity):
  - Linear search: TRLM-FoBa (reverse) 53.48, TRLM-Ba 53.16, TRLM-Fo 45.00, Forward Baseline 9.33
  - Binary search: TRLM-Ba 45.09, TRLM-Fo 43.14, Forward Baseline 5.88
  - Exclusion search: TRLM-FoBa (forward) 38.67, TRLM-Fo 37.90, TRLM-Ba 36.33, Forward Baseline 4.66
  - The introduction states a 44.15% improvement in citation attribution accuracy over the forward baseline
- **Passage retrieval**:
  - MS-MARCO NDCG@10: TRLM-Ba 61.49, TRLM-FoBa (reverse) 58.84, TRLM-Fo 54.46, Forward Baseline 53.00
  - NF-Corpus NDCG@10: TRLM-Fo 48.08, TRLM-FoBa (forward) 48.00, TRLM-Ba 43.23, Forward Baseline 3.89
  - The introduction states an NDCG@10 improvement of 44.13 points on NF-Corpus
- **Defence** (false negative rate on a human-annotated toxic set "HA" and on JailbreakBench "JBB"; false positive rate on hard and easy safe questions):
  - The gpt-3.5 input filter alone has a false negative rate of 25.58% on the human-annotated set
  - Threshold 2, pre-trained TRLM-Ba: FNR 18.18 (HA) / 52.78 (JBB), FPR 0.00 (hard) / 8.00 (easy)
  - Threshold 2, pre-trained TRLM-Fo: FNR 0.00 (HA) / 36.11 (JBB), FPR 17.00 (hard) / 2.00 (easy)
  - The abstract states a large reduction in false negative rate with a negligible effect on false positive rate

### Ablation Studies

- The comparison of the variants is the ablation: prompting in reverse only (TRLM-Fo) against training in reverse (TRLM-Ba) against the two orders (TRLM-FoBa)
- TRLM-Ba is the best on AlpacaEval with Gemini and on MS-MARCO. It is not the best in all tables (Mixtral reranking, NF-Corpus).
- Pre-trained against instruction-tuned (FLAN) TRLM in the defence task
- Thresholds 2, 4, 6 in the defence task: a higher threshold gives a higher false negative rate and a lower false positive rate

### Efficiency analysis

- Citation attribution: linear search and exclusion search need O(N) scoring calls, binary search needs O(log N) calls
- No other efficiency analysis

## Preliminaries

### Best-of-N reranking

- Sample N responses from the generator
- Score each response with a scorer model
- Return the response with the best score
- The scorer is the only source of feedback, so its quality decides the gain

### Bayes view of reverse scoring

`P(A|Q) ∝ P(Q|A) * P(A)`

- A forward model gives `P(A|Q)`. A reverse model gives `P(Q|A)`.
- `P(Q|A)` asks: "if I see this answer, how sure am I that it came from this question?"
- A generic or off-topic answer is compatible with many questions, so `P(Q|A)` is low for it
- The paper's intuition for reverse order: introductions follow conclusions, questions follow answers

### KL-constrained alignment

```python
# find a new policy that gets a high reward but stays near the old one
maximize  E_new[reward(Q, A)] - (1 / alpha) * KL(P_new || P_old)
# the solution has the form
P_new(A|Q) ∝ P_old(A|Q) * exp(alpha * reward(Q, A))
```

- If `reward = log P_old(A|Q)`, the solution is `P_old ^ (1 + alpha)`. This is only a temperature change (Lemma 1).
- If `reward = log P_reverse(Q|A)`, the solution is `P_old * P_reverse ^ alpha` (Lemma 2).

## GPU hours

- Pre-training of the TRLM models: two TPUv5e pods for two weeks
- FLAN fine-tuning: one TPUv5e pod for one day
- GPU hours: Not mentioned

## Key takeaways

- The reverse direction (response -> query) is real, new feedback. Forward self-scoring is only temperature scaling.
- Reverse training (TRLM-Ba) is usually better than reverse prompting of a forward model (TRLM-Fo), but not always
- The direction matters most when one side is short and the other is long. To score a long, complex document with a short query in the forward direction is poor (NF-Corpus: 3.89 against 48.08 NDCG@10).
- Reverse generation is useful also: generate the possible questions of an answer, then check those questions with a safety filter
- The gain on AlpacaEval is up to 5% over self-scoring
- The work uses the reverse model only for reranking at inference. It does not use the reverse score as a training reward (the authors list this as future work).

## What I still do not understand?

## Ideas to pursue

- **In common with the causal-consistency objective:** Both use the reverse direction as an unsupervised check on the forward direction. Here `P(query | response)` checks `P(response | query)`. In the idea, "Therefore?" (cause -> line) checks "Why?" (line -> cause). Both need no labels and both need only raw corpus text for training.
- **What it lacks:** The reverse direction is token order (time), not causal direction. There is no cycle: the model never generates in one direction and reconstructs in the other. The reverse score is used only at inference to rerank; it is not a loss, so the forward model does not become more consistent. It works on one (query, response) pair and has no cross-document link.
- **What to borrow:** (1) Lemma 1 as a warning - if the same forward model generates the "Why?" answer and then scores the original line, the signal can collapse to temperature scaling. The distractor option, or a separate direction, must add new information. (2) Lemma 2 as a design - the product `P_forward * P_reverse ^ alpha` is a simple way to combine the two directions. (3) The reverse score `P(original line | abduction)` as a cheap filter or as a hard-distractor miner: pick distractor lines that the reverse model scores high.

## Similar papers

- [Deductive Closure Training of Language Models for Coherence, Accuracy, and Updatability](https://arxiv.org/abs/2401.08574)
- [RevThink: Reverse Thinking Makes LLMs Stronger Reasoners](https://arxiv.org/abs/2411.19865)
- [Reasoning to Learn from Latent Thoughts](https://arxiv.org/abs/2503.18866)
- [Cycle-Instruct: Fully Seed-Free Instruction Tuning via Dual Self-Training and Cycle Consistency](https://arxiv.org/abs/2508.16100)
- [Flip-Flop Consistency: Unsupervised Training for Robustness to Prompt Perturbations in LLMs](https://arxiv.org/abs/2510.14242)
- [LEDOM (ACL 2026)](https://aclanthology.org/2026.acl-long.975)
- [The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"](https://arxiv.org/abs/2309.12288)
- [Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations](https://arxiv.org/abs/2205.11822)
