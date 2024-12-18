# [Instruction Mining: Instruction Data Selection for Tuning Large Language Models](https://openreview.net/forum?id=wF6k0aWjAu)

## Meta

* Journal   - COLM
* Year      - 2024
* Author    - linkedin
* Code      - https://openreview.net/forum?id=wF6k0aWjAu (see supplementary)
* Slides    - 
* One liner - Perform linear regression between validation loss after finetuning on the shortlist and off the shelf indicators
* Model     - 
* Datasets  - ALPACA, OPEN-ASSISTANT, STACKEXCHANGE, WIKIHOW, DOLLY, OPENORCA
* Baselines - VICUNA-1.5-7B, LLAMA-2-7B-chat, LLAMA-2-7B, STABLEBELUGA-7B 

## Flow

* Generate datasets from existing datasets by random subsampling and merging and ensure each set contains 1,000 examples.
* Fine-tune a base language model (LLAMA-2-7B) on each sampled dataset for three epochs.
* Evaluate each fine-tuned model on a shared evaluation set (SELF-INSTRUCT or MT-BENCH). Record validation loss.
* Compute indicators for the dataset using the metrics given below
* Do a regression fit of validation loss vs indicators
* Use the regression to shortlist unseen datasets
* Use the shortlist to finetune the model and validate on the unseen datasets

## Equations

## Proofs

## Algorithms

### Indicators

* **knn6**    - Cosine similarity of top 6 nearest neighbors from some DB dataset
* **length**  - Number of tokens
* **mtld**    - Rare word usage. `from lexicalrichness import LexicalRichness`
* **ppl**     - Perpexity of `meta-llama/Llama-2-7b-hf` on the generated text
* **pythia**  - Use an existing RLHF reward model `OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5`
* **unieval** - Use scores returned by the [Unieval](https://arxiv.org/abs/2210.07197) model

## Experiments

* Figure 1,4: double descent? Maybe increasing the subset selected data size increases more irrelevant entries? Also, it is finetuned for constant epochs, not steps, that might explain the second descent, as further on X-axis the train time/compute also increases.
* Regression scatter
* Indicator distributions

## Rebuttals

### Reviewer 1

* Least squares - Ablation to combine indicators instead of linear independence assumption.

### Reviewer 2

* More models (e.g., Mistral)
* not very novel
* "double descent phenomenon is somewhat as expected"
* "linear combination seems overly simple"
* "it applies an existing method BLENDSEARCH"

### Reviewer 3

* Lacks enough baselines
