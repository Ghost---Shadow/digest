# [Reasoning to Learn from Latent Thoughts](https://arxiv.org/abs/2503.18866)

## Meta

* Journal   - Not mentioned (arXiv preprint)
* Year      - 2025
* Author    - Stanford University, University of Toronto, Vector Institute (Ruan, Band, Maddison, Hashimoto)
* Code      - [ryoungj/BoLT](https://github.com/ryoungj/BoLT)
* Slides    - 
* One liner - Treat web text as the compressed result of hidden thoughts. Infer the thoughts, train on thought plus text in both orders, and bootstrap with an EM loop (BoLT).
* Model     - TinyLlama-1.1B
* Datasets  - FineMath-4+ (training); MATH, GSM8K, MMLU-STEM (evaluation); DCLM validation set (general-domain NLL)
* Baselines - Raw-Repeat, Raw-Fresh, WRAP-Orig, WRAP-CoT

## What problem does it solve?

- Compute for pre-training grows faster than the quantity of human-written text. Data will become the bottleneck
- Web text is the compressed final result of a longer human thought process. The thoughts are not in the text
- If the model also learns the hidden thoughts behind the text, it can learn more from each raw token (data efficiency)

## How does it solve it?

### Training flow

Learning from latent thoughts

1. Split each document into chunks `X_n` at sentence boundaries (Poisson length, mean 8 sentences, limited to 1-20)
2. For each chunk, get a latent thought `Z_n` in natural language. `Z_n` gives the missing background knowledge and the step-by-step reasoning behind `X_n`. The context is the 3 chunks before
3. At the start, a surrogate model (GPT-4o-mini) writes the latents. The prompt gives a prefix and a suffix and asks for the thoughts that go between them
4. Put the latent between the special tokens `<StartOfLatent>` and `<EndOfLatent>`
5. Flip a coin for each chunk to select the format:
   - `<Prior>` mode: latent before the chunk. This trains the joint `p(Z, X)`
   - `<Post>` mode: latent after the chunk. This trains the posterior `q(Z | X)`
6. Train one autoregressive LM on this text with standard next-token prediction. The same weights are the joint model and the posterior model

BoLT (Bootstrapping Latent Thoughts), an EM loop

1. Warmstart: train `M_0` on a small dataset with GPT-4o-mini latents
2. E-step: for each chunk `X_n` with context `C_n`, sample `K` latents from the posterior of the current model `M_{t-1}`
3. Compute an importance weight for each latent: joint probability divided by posterior probability
4. Resample one latent in proportion to the weights
5. M-step: train `M_t` on the data with the new latents (from scratch in the fixed-data setup, or continued from `M_{t-1}` in the continual setup)
6. Repeat. No task-specific data is used

### Inference

1. Use few-shot chain-of-thought prompts
2. Put the reasoning of each example inside the latent tokens with the `<Prior>` prefix. The model then reasons in the latent `Z`, not in the visible text
3. The prediction is a sample from the marginal `E_{p(Z|X)}[p(X' | Z, X)]`

### Equations

#### Model

`p(Z, X) = p(Z) · p(X | Z)`

- `X` - observed text chunk
- `Z` - latent thought (natural language)
- `q(Z | X)` - approximate posterior, "decompress" the text into thoughts

#### ELBO

`ELBO = E_{Z~q(Z|X)}[log p(X|Z) + log p(Z)] + H[q(Z|X)]`

`ELBO = E_{Z~q(Z|X)}[log p(X|Z)] − KL[q(Z|X) || p(Z)]`

- `log p(X|Z)` - utility: the thought must help to predict the text
- `log p(Z)` - simplicity: the thought must be easy to predict
- `H[q(Z|X)]` - diversity of the thoughts
- The paper uses the ELBO as a diagnostic. It is a loose bound on the NLL

#### E-step with importance resampling

```python
for X_n, C_n in chunks:
    Z = [sample(q, X_n, C_n, model=M_prev) for k in range(K)]
    w = [p(Z[k], X_n, given=C_n, model=M_prev) / q(Z[k], given=(X_n, C_n), model=M_prev)
         for k in range(K)]
    j = categorical(w)        # resample one latent in proportion to the weights
    Z_n = Z[j]
```

- `K` - number of Monte Carlo samples (default 4)
- `w[k]` - importance weight of latent k
- When `K` increases, the importance-weighted bound becomes tighter and the resampled posterior moves nearer to the true posterior

### Model

- TinyLlama-1.1B (not trained on math data before)
- AdamW, `β1 = 0.9`, `β2 = 0.95`, weight decay 0.01
- Cosine schedule, 1000 warmup steps, peak learning rate 1e-4 (tuned)
- Sequence length 2048, batch size 96
- Synthetic-latent experiment: 480M raw tokens + about 1.1B synthetic tokens, total budget 8B training tokens
- Bootstrapping: batch size 192, latent sampling temperature 1.0, `K = 4`, 2 epochs for each iteration
- Continual bootstrapping: 4 iterations, constant learning rate 3e-5 with linear decay in the last 15% of each stage

## How is this paper novel?

- Sees pre-training as inference of latent thoughts: the model reasons to learn from data
- Joint training of the latent and the text with only next-token prediction and special tokens. No change to the architecture
- One model is both the generator of thoughts and the learner, so a better model writes better thoughts for the next round
- BoLT: an EM procedure with Monte Carlo importance resampling. A 1.1B model improves its own latents through several iterations with no task-specific data
- Shows that more inference compute in the E-step (larger `K`) gives better training data

## List of experiments

- **Synthetic latents, data-constrained (MATH / GSM8K / MMLU-STEM)**
  - Raw-Repeat (repeat the 480M raw tokens, about 16 epochs): 5.74 / 5.76 / 27.31
  - Raw-Fresh (8B unique raw tokens): 11.18 / 13.27 / 30.63
  - WRAP-Orig (paraphrases): 11.06 / 12.43 / 31.40
  - WRAP-CoT (paraphrases with step-by-step reasoning): 19.36 / 21.08 / 34.51
  - Latent Thought: 25.38 / 33.59 / 35.87
- **Bootstrapping on fixed data (1.92B raw tokens)**
  - Few-shot MATH accuracy improves through iteration 3. Fine-tuned MATH gains continue through iteration 4
  - ELBO and NLL improve through iteration 4
  - Better than a FLOP-matched raw baseline and a raw-token-matched baseline
  - Negative result: few-shot GSM8K accuracy decreases with more iterations (still above the raw baselines). The authors think the cause is weaker general language understanding after math-heavy training (the DCLM NLL gets worse)
- **Continual bootstrapping (corpus grows to 3.84B raw tokens)**
  - The latest bootstrapped generator gives a steeper scaling curve than a generator fixed at an early iteration
  - The gap in MATH accuracy becomes larger in later iterations
  - Results are in figures only, no numbers in the text

### Ablation Studies

- **Mix latent data 1:1 with raw text that has no latents (MATH / GSM8K / MMLU-STEM):** 22.38 / 20.17 / 33.33. Worse than the full method
- **Chain-of-thought in raw text space at evaluation, not in latent space:** 20.34 / 22.97 / 31.78. Worse than the full method
- **Number of Monte Carlo samples `K = 1, 2, 4, 8`:** ELBO and MATH accuracy improve monotonically with `K`
- **Few-shot prompt type:** synthetic GPT-4o-mini CoT prompts are better than the standard CoT prompts for all methods
- **WRAP mixing ratio (0.0, 0.25, 0.5 raw):** fully synthetic data (0.0) is best for the two WRAP baselines

### Efficiency analysis

- The method sees each raw token 3.3 times fewer than the baselines at the same training-token budget, because the latent-to-raw token ratio is about 2.3
- With 480M raw tokens it is better than Raw-Fresh, which uses 8B unique raw tokens
- About 13B training tokens for each bootstrap iteration
- The E-step is extra inference compute: `K` samples for each chunk

## Preliminaries

### Latent variable model and ELBO

- We want to maximize `log p(X)`, but `p(X) = Σ_Z p(Z) p(X|Z)` is intractable (too many possible `Z`)
- Use a helper distribution `q(Z|X)` and maximize a lower bound:

`log p(X) >= E_{Z~q}[log p(Z, X) − log q(Z|X)]`

- The bound is tight when `q(Z|X)` is equal to the true posterior `p(Z|X)`

### Expectation-Maximization (EM)

1. E-step: with the current model, infer the latent `Z` for each data point `X`
2. M-step: with the inferred `Z`, train the model to maximize the likelihood of `(Z, X)`
3. Repeat. Each step can only make the bound better or keep it the same

### Importance weighting (IWAE, Burda et al. 2015)

- Draw `K` samples from `q` and not one
- Weight for each sample: `w_k = p(Z_k, X) / q(Z_k | X)`
- A sample has a high weight if the joint model likes it more than the posterior expected
- The bound with `K` samples is tighter than the ELBO, and becomes tighter as `K` increases:

```python
bound_K = log(mean([p(Z[k], X) / q(Z[k], given=X) for k in range(K)]))
```

- Resampling one `Z` in proportion to `w` gives a sample from a better posterior than `q`

## GPU hours

- Total GPU hours: Not mentioned
- Hardware: 4x H200 for the synthetic-latent training, 8x H200 for the bootstrapping training
- Latent generation: a distributed cluster of mixed GPUs (H100/H200, A100/A6000/A5000/A40, L40, RTX 3090)

## Key takeaways

- Text with the reasoning behind it is better training data than more raw text
- The latents must be paired with the text and kept in a separate latent space. Mixing with unpaired raw text or reasoning in plain text space loses most of the gain
- The model can bootstrap: a model trained on thoughts writes better thoughts for the next model
- Selection among `K` samples by importance weight is a cheap quality filter. More samples give better data
- Limits stated by the authors: one small model (1.1B), mostly math data, each latent is local to one chunk ("myopic", no high-level plan), and few-shot GSM8K gets worse with more bootstrap iterations

## What I still do not understand?

## Ideas to pursue

- **In common with the causal-consistency objective:** the "Why?" answer is the same object as the latent thought `Z`: text that the model abduces from a corpus line `X` to explain it. `<Post>` mode (`X -> Z`) is the abduction step, and `<Prior>` mode (`Z -> X`) is the "Therefore?" step. Both use only next-token prediction on raw pre-training lines, and both use the same model for the two directions
- **What BoLT lacks:** the cycle is generative, not discriminative. There is no distractor line and no multiple-choice loss, so nothing makes the thought specific to its own line against a different line. The latent is conditioned only on the local context (3 chunks before) of the same document, so there is no explicit consistency across documents. Warmstart needs an external model (GPT-4o-mini)
- **What the idea can borrow:**
  - The importance weight as a filter for the "Why?" answers. Sample `K` answers, score each with `p(answer, line) / q(answer | line)`, and keep one. This is a principled form of "the answer must make the original line more probable"
  - The EM view. The abduction step is the E-step and the "Therefore?" loss is the M-step. Repeat the loop so better abductions give better training data
  - The special tokens and the coin flip between `<Prior>` and `<Post>` formats, so one model learns the two directions with no change to the pipeline
  - The warning from the ablations: keep the abduced text separate from normal text, and watch general-domain NLL for degradation after several rounds

## Similar papers

- [Deductive Closure Training](https://arxiv.org/abs/2401.08574)
- [RevThink: Reverse Thinking Makes LLMs Stronger Reasoners](https://arxiv.org/abs/2411.19865)
- [Cycle-Instruct](https://arxiv.org/abs/2508.16100)
- [Time-Reversal Provides Unsupervised Feedback to LLMs](https://arxiv.org/abs/2412.02626)
- [Flip-Flop Consistency](https://arxiv.org/abs/2510.14242)
- [LEDOM](https://aclanthology.org/2026.acl-long.975)
- [The Reversal Curse](https://arxiv.org/abs/2309.12288)
- [Maieutic Prompting](https://arxiv.org/abs/2205.11822)
