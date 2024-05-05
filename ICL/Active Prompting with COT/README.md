# [Active Prompting with Chain-of-Thought for Large Language Models](https://arxiv.org/pdf/2302.12246)

## Meta

* Journal - (rejected from ICLR24) https://openreview.net/forum?id=wabp68RoSP
* Year - 2023
* Author - The Hong Kong University of Science and Technology, University of Toronto, The University of Hong Kong
* Code - https://github.com/shizhediao/active-prompt
* One liner - spicy least confidence subset selection in a chain of thought setting
* Model - UL2-20B, LaMDA-137B, PaLM 540B, text-davinci-002, code-davinci-002, text-davinci-003
* Datasets - GSM8K, ASDiv, SVAMP, AQuA, SingleEq, CSQA*, StrategyQA*, Letter
* Baselines - Cot, self-consistency, AutoCot, randomCot, 

## Training flow

## Equations

To determine which questions to annotate, the method uses uncertainty estimation as a metric to select the most uncertain questions. This involves multiple strategies:
1. **Disagreement**: The uncertainty is assessed based on the variety of answers produced for each question by running the LLM multiple times. A higher number of unique answers suggests greater uncertainty.
2. **Entropy**: This metric calculates the entropy of the answer frequencies, with higher entropy indicating greater uncertainty.
3. **Variance**: Particularly thought to be effective for numerical answers, variance measures the spread of numerical predictions, which is normalized to prevent domination by larger numbers.
4. **Self-Confidence**: This involves the LLM assessing its own confidence in its predictions, selecting the least confident answers.

Baselines

- Chain-of-thought (CoT) (Wei et al., 2022b): standard chain-of-thought prompting which provides four to eight human-written exemplars consisting of a series of intermediate reasoning steps.
- Self-consistency (SC) (Wang et al., 2022c): an improved version of CoT. Instead of greedy decoding, it samples a set of reasoning paths and chooses the most common answer.
- Auto-CoT (Zhang et al., 2022b): an automatic exemplar construction method by clustering and generating rationales with zero-shot prompting (Kojima et al., 2022).
- Random-CoT: a baseline of Active-Prompt. It shares the same annotation process as Active-Prompt. The only difference is that it randomly samples questions from the training data for annotation instead of applying our proposed uncertainty metrics.

## Algorithms

1. quantify uncertainity using the 4 methods above
2. pick top-n most uncertain questions to annotate

## Experiments

* big table
* accuracy vs num predicted answers (number of branches of thought?)
