# Reward-Oriented Data Selection Framework for Instruction Tuning

## Meta

* **Name**: ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning
* **Journal**: Under review at International Conference on Learning Representations (ICLR) 2025
* **Year**: 2025
* **Author**: Anonymous (under double-blind review)
* **Code**: Will be made publicly available upon acceptance
* **One-liner**: ROSE utilizes pairwise preference loss to improve task-specific instruction tuning of LLMs.
* **Model**: Llama (LLAMA-2-7B, LLAMA-2-13B, LLAMA-3.1-8B, LLAMA-3.1-8B-INS.), Mistral (MISTRAL-7B-V0.3, MISTRAL-7B-INS.-V0.3)
* **Datasets**: DOLLY, OPEN ASSISTANT 1, FLAN V2, COT; Evaluation: SHP, Stack Exchange (SE), HH-RLHF
* **Baselines**: Random, BM25, Representation-based data selection (RDS), DSIR, Influence Functions, LESS, Shapley values

## Formulas

### Cross-Entropy Loss for Instruction Tuning

The cross-entropy loss is represented as:

$$
\text{Cross-Entropy Loss} = -\sum \text{target} \; \log(\text{prediction})
$$

#### Components:
- **target**: True token or one-hot encoded vector indicating the correct token for prediction.
- **prediction**: Model’s predicted probability for the correct token.
- **sum**: Summation over all tokens in a sequence.

### Influence Estimation Scheme

Change in validation loss from parameter update is:

$$
L(D_{\text{val}}; \theta_t) - L(D_{\text{val}}; \theta_{t-1}) = \langle \nabla_\theta L(D_{\text{val}}; \theta_{t-1}),\, \delta \theta \rangle
$$

#### Components:
- **$L(D_{\text{val}}; \theta)$**: Loss on validation dataset with model parameters $\theta$.
- **$\theta_{t-1}$ and $\theta_{t}$**: Model parameters before and after an update.
- **$\nabla_\theta L(D_{\text{val}}; \theta_{t-1})$**: Gradient of validation loss w.r.t parameters at $\theta_{t-1}$.
- **$\delta \theta$**: Change in model parameters, defined as:

  $$
  \delta \theta = -\alpha \cdot \nabla_\theta L(z; \theta_{t-1})
  $$

- **Inner Product $\langle \cdot, \cdot \rangle$**: Describes alignment between parameter change and validation loss gradient.

### Reward-Oriented Optimization Framework

ROSE employs a reward function defined by:

$$
r(x, y) = \beta \log \!\left( \frac{\Omega_\theta(y \mid x)}{\Omega_{\text{ref}}(y \mid x)} \right) + \beta \log Z(x)
$$

#### Components:
- **$r(x, y)$**: Reward for output $y$ given input $x$.
- **$x, y$**: Input prompt and generated output.
- **$\beta$**: Scaling factor for reward magnitude.
- **$\Omega_\theta(y \mid x)$ and $\Omega_{\text{ref}}(y \mid x)$**: Model and reference probability scores.
- **$Z(x)$**: Partition function for normalizing probabilities.

The optimization loss within ROSE is given by:

$$
L_{\text{ROSE}}(\Omega_\theta; \Omega_{\text{ref}}) = -\mathbb{E}_{(x,y^w,y^l)\sim D'_{\text{val}}} \!\left[ \log \sigma\!\left( \beta \log \frac{\Omega_\theta(y^w \mid x)}{\Omega_{\text{ref}}(y^w \mid x)} - \beta \log \frac{\Omega_\theta(y^l \mid x)}{\Omega_{\text{ref}}(y^l \mid x)} \right) \right]
$$

#### Gradient for ROSE loss:

$$
\nabla_\theta L_{\text{ROSE}}(\Omega_\theta; \Omega_{\text{ref}}) = - \beta \, \mathbb{E}_{(x, y^w, y^l)\sim D'_{\text{val}}} \!\left[ \sigma\left(\hat{r}_\theta(x, y^l) - \hat{r}_\theta(x, y^w)\right) \cdot \left( \nabla_\theta \log \Omega_\theta(y^w \mid x) - \nabla_\theta \log \Omega_\theta(y^l \mid x) \right) \right]
$$

## Training Flow

### Training Steps

1. **Model Initialization**
   - Start with models $ \Omega $ and the final model $ \Gamma $.
   - Initialize $ \Omega $ with a random 5% subset of data $ D $.

2. **Preference Validation Set Transformation**
   - Convert validation data $ D_{\text{val}} $ into preference validation set $ D'_{\text{val}} $.

3. **Gradient Calculation**
   - Perform initial backpropagation on $ \Omega $.
   - Compute gradients for training data $ D $ and preference set $ D'_{\text{val}} $.

4. **Influence Score Evaluation**
   - Evaluate influence scores for each data point.

5. **Data Selection**
   - Select top 5% data points from $ D $ with highest influence scores.

6. **Final Model Training**
   - Train final model $ \Gamma $ on selected dataset $ D_{\text{train}} $.

7. **Output**
   - Fine-tuned instruction LLM $ \Gamma' $ for task-specific goals.

## Inference Flow

### Inference Steps

1. Transform validation set into a preference set for task performance.
2. Extract gradients for training and validation data.
3. Calculate influence scores relative to the validation set.
4. Select top 5% of data points with highest influence scores for final tuning.
5. Train the final model on the selected subset to optimize reward.

## Experiments

### List of Experiments
* Comparison with baselines on various datasets (Table 1)
* ROSE performance across model architectures (Table 2)
* Validation Loss vs Test Win Rate correlation (Figure 3)
* Data selection impact by number of checkpoints (Table 3)
* Performance comparison across validation shots (Figure 4)
* Transfer ability analysis on different datasets and models (Table 9)
* Subtask results across benchmarks (Tables 10, 11, 12)

## Proofs

### List of Proofs

1. **Submodularity of Influence Function**: Analysis of gradient-based influence estimation exhibiting submodularity in data selection.
2. **Lower Bound Termination Guarantee**: Framework ensures data subset selection for maximized reward, terminating with established lower bounds.
3. **Time Complexity Analysis**: Discussion of computational aspects to show feasible polynomial time complexity through empirical validation.