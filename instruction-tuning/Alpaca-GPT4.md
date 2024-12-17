# [Instruction Tuning with GPT-4](https://arxiv.org/abs/2304.03277)

## Meta

* Journal   - Arxiv
* Year      - 2024
* Author    - microsoft
* Code      - https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
* Slides    - 
* One liner - Train a reward model with RLHF, generate lots of instructions and pick the best ones using the reward model.
* Model     - LLaMA 7B
* Datasets  - Alpaca dataset 
* Baselines - 

## Flow

### Training: Surrogate Model (Reward Model)
1. Data Generation: Collect responses from various models to the same prompts.
2. Pairing Responses: Create pairs of responses for each prompt.
3. Human Judgments: Gather human evaluations for these pairs to determine the better response.
4. Training the Reward Model: Train the model to mimic human judgments.
5. Model Design: Use a binary classifier or a regression model to compute scores for each response.
6. Optimization: Refine model parameters to align predicted scores with human preferences.

### Inference: Iterative Instruction Generation and Evaluation
1. Instruction Generation: Use a generative model or human input to create new instructions.
2. Using the Reward Model: Apply the reward model to evaluate responses to new instructions.
3. Iterative Refinement: Refine or generate new instructions based on model evaluations.
4. Goal: Enhance the quality and relevance of instructions and responses continuously.

## Experiments

* bigram frequency
* output length frequency
* LLM for reward model vs histogram of scores
* HHH human evaluation
* Win-rate between models
* Response length vs rougeL for unnatural instructions dataset
* survey form for humans
