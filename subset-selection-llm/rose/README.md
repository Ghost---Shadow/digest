# ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning

## Meta

- **Name**: ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning
- **Journal**: Under review at International Conference on Learning Representations (ICLR) 2025
- **Year**: 2025
- **Author**: Anonymous (under double-blind review)
- **Code**: To be made publicly available upon acceptance
- **One Liner**: A novel data selection method, ROSE, leverages pairwise preference loss for enhancing task-specific instruction tuning of LLMs.
- **Model**: Llama (LLAMA-2-7B, LLAMA-2-13B, LLAMA-3.1-8B, LLAMA-3.1-8B-INS.), Mistral (MISTRAL-7B-V0.3, MISTRAL-7B-INS.-V0.3)
- **Datasets**: DOLLY, OPEN ASSISTANT 1, FLAN V2, COT; Evaluation: SHP, Stack Exchange (SE), HH-RLHF
- **Baselines**: Random, BM25, Representation-based data selection (RDS), DSIR, Influence Functions, LESS, Shapley values

## Formulas

This section provides a detailed explanation of the variables within the presented formulas using MathJax-style LaTeX notation.

### Instruction Tuning Loss

The instruction tuning loss is commonly defined using the cross-entropy loss for next-token prediction. It can be formalized as follows:

\[
L_{\text{IT}}(\theta) = - \sum_{t} \log p_\theta(w_t | w_{<t})
\]

Where:
- \(\theta\) denotes model parameters.
- \(w_t\) is the token at position \(t\).
- \(w_{<t}\) represents all tokens preceding \(w_t\).
- \(p_\theta(w_t | w_{<t})\) is the predicted probability of token \(w_t\) based on the preceding tokens.

The objective is to optimize \(\theta\) to minimize \(L_{\text{IT}}\).

### Pairwise Preference Loss

Rather than solely predicting the next token, the pairwise preference loss exploits labeled preferences between different responses to prioritize preferred (or "winning") responses over less-favored ones. This methodology is foundational to the demonstrated formulas.

### Influence of Training Data on Validation Loss

A first-order Taylor expansion is employed to determine which training samples are most pertinent for a specific validation set \(D_{val}\):

\[
L(D_{val}; \theta_t) - L(D_{val}; \theta_{t-1}) = \langle \nabla_\theta L(D_{val}; \theta_{t-1}), \delta\theta \rangle
\]

Where:
- \(L(D_{val}; \theta)\) is the validation set loss using parameters \(\theta\).
- \(\theta_{t-1}\) and \(\theta_t\) are parameter values before and after an update.
- \(\nabla_\theta L(D_{val}; \theta_{t-1})\) represents the gradient of the validation loss with respect to \(\theta\).
- \(\delta\theta\) signifies the parameter change due to the update.
- \(\langle \cdot, \cdot \rangle\) denotes the inner product.

The parameter update using Stochastic Gradient Descent (SGD) is given by:

\[
\delta\theta = -\alpha \cdot \nabla_\theta L(z; \theta_{t-1})
\]

Where:
- \(\alpha\) is the learning rate.
- \(z\) is a single training data sample.
- \(\nabla_\theta L(z; \theta_{t-1})\) is the gradient of the loss for sample \(z\) with respect to \(\theta\).

The Taylor expansion becomes:

\[
L(D_{val}; \theta_t) - L(D_{val}; \theta_{t-1}) \propto \langle \nabla_\theta L(D_{val}; \theta_{t-1}), \nabla_\theta L(z; \theta_{t-1}) \rangle
\]

### ROSE Formulation

The ROSE method introduces a reward-based framework for data selection and instruction tuning.

#### a) Reward Function

\[
r(x, y) = \beta \log \frac{\Omega_\theta(y \mid x)}{\Omega_{\text{ref}}(y \mid x)} + \beta \log Z(x)
\]

Where:
- \(r(x, y)\) represents the reward for generating output \(y\) for input \(x\).
- \(\beta\) is a scaling factor.
- \(\Omega_\theta(y \mid x)\) is the probability of \(y\) given \(x\) under the current model (\(\theta\)).
- \(\Omega_{\text{ref}}(y \mid x)\) denotes the probability under a reference model.
- \(Z(x)\) is a normalizing constant.

#### b) ROSE Optimization Objective

\[
L_{\text{ROSE}}(\Omega_\theta; \Omega_{\text{ref}}) = -\mathbb{E}_{(x,y_w,y_l)\sim D'_{val}} \left[\log \sigma \left(\beta \log \frac{\Omega_\theta(y_w \mid x)}{\Omega_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\Omega_\theta(y_l \mid x)}{\Omega_{\text{ref}}(y_l \mid x)}\right)\right]
\]

Where:
- \(L_{\text{ROSE}}(\Omega_\theta; \Omega_{\text{ref}})\) is the ROSE method's loss function.
- \(\mathbb{E}_{(x,y_w,y_l)\sim D'_{val}}[\cdot]\) denotes expectation over the preference validation distribution \(D'_{val}\).
- \(\sigma(\cdot)\) is the logistic sigmoid function.
- The expression within the sigmoid represents the log-probability ratio difference for winning and losing outputs.

The minimized objective ensures higher winning sample preference under the current model.

#### c) Gradient of the ROSE Loss

\[
\nabla_\theta L_{\text{ROSE}}(\Omega_\theta; \Omega_{\text{ref}}) = -\beta \, \mathbb{E}_{(x,y_w,y_l)\sim D'_{val}} \left[\sigma\big(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\big) \left(\nabla_\theta \log \Omega(y_w \mid x) - \nabla_\theta \log \Omega(y_l \mid x)\right)\right]
\]

Where:
- \(\nabla_\theta L_{\text{ROSE}}(\Omega_\theta; \Omega_{\text{ref}})\) is the ROSE loss gradient concerning \(\theta\).
- \(\hat{r}_\theta(x, y)\) is the reward under current parameters \(\theta\).
- The expression guides parameter adjustments to favor winning outputs.

### Summary

- **Instruction tuning** focuses on cross-entropy loss minimization for next-token prediction.
- The **pairwise preference** approach emphasizes comparative judgments rather than absolute predictions.
- Taylor expansion connects validation loss changes to validation-training gradient inner products, aiding significant sample selection.
- The **ROSE formulation** combines reward-based log-probability ratios for data selection, driving preferences in parameter optimization.

## Training Flow

### Training Process

1. **Model Initialization**
   - Begin with models \( \Omega \) and \(\Gamma\).
   - Select a random 5% data subset for initial training.

2. **Preference Validation Set Transformation**
   - Convert the few-shot SFT validation data set into a preference validation set.

3. **Gradient Calculation**
   - Perform initial backpropagation, compute Adam gradients for training corpus \( D \), and SGD gradients for the preference validation set.

4. **Influence Score Evaluation**
   - Calculate each data point's influence score using gradient-based assessment.

5. **Data Selection**
   - Select the top 5% data points based on influence scores.

6. **Final Model Training**
   - Train the final model \( \Gamma \) using selected data.

7. **Output**
   - Obtain the task-optimized LLM.

### Training Pseudocode

```python
# High-level pseudocode

# Step 1: Initial Data Selection and Training
initialize_model(Omega)
random_subset = random_select(D, percent=0.05)
initial_train(Omega, random_subset)

# Step 2: Preference Validation Set
D_val_prime = transform_to_preference(D_val)

# Step 3: Gradient Computation
Omega_prime = backpropagate(Omega) # updates weights
gradients_train = compute_gradients(Omega_prime, D, method='Adam')
gradients_val = compute_gradients(Omega_prime, D_val_prime, method='SGD')

# Step 4: Influence Score Evaluation
influence_scores = calculate_influence(gradients_train, gradients_val)

# Step 5: Data Selection
D_train = select_top_influence(D, influence_scores, top_percent=0.05)

# Step 6: Final Training
train(Gamma, D_train)

# Final Model Output
Gamma_prime = get_finetuned_model()
```

## Inference Flow

### Inference Process

1. Transform the validation set into a preference set representing task performance.
2. Backpropagate and extract data point gradients.
3. Compute each data point's influence score using calculated gradients.
4. Select the top 5% data points with highest influence scores.
5. Train the final model on this data subset for task-specific performance maximization.

### Inference Pseudocode

```python
# Step 1: Transform validation set to preference set
def transform_to_preference_set(D_val):
    D_val_preference = [] 
    for prompt, response in D_val: 
        # Generate additional responses and evaluate 
        winning_response, losing_response = evaluate_responses(prompt, response) 
        D_val_preference.append((prompt, winning_response, losing_response)) 
    return D_val_preference

# Step 2: Backpropagate and extract gradients
def extract_gradients(model, D, D_val_preference):
    # Initialize lists for storing gradients
    training_gradients = []
    validation_gradients = []

    # Compute gradients for each data point
    for data_point in D:
        gradient = model.compute_gradient(data_point)
        training_gradients.append(gradient)

    for prompt, win_res, loss_res in D_val_preference:
        gradient = model.compute_preference_gradient(prompt, win_res, loss_res)
        validation_gradients.append(gradient)
    
    return training_gradients, validation_gradients

# Step 3: Compute influence scores
def compute_influence_scores(training_gradients, validation_gradients):
    influence_scores = []
    for train_grad in training_gradients:
        score = sum(np.dot(val_grad, train_grad) for val_grad in validation_gradients)
        influence_scores.append(score)
    return influence_scores

# Step 4: Select the top data points
def select_top_data(D, influence_scores, top_percentage=0.05):
    num_top_points = int(len(D) * top_percentage)
    top_indices = np.argsort(influence_scores)[-num_top_points:]
    return [D[i] for i in top_indices]

# Step 5: Final model training
def train_final_model(selected_data, model):
    model.train_on(selected_data)
    return model

# Inference process
D_val_preference = transform_to_preference_set(D_val)
training_gradients, validation_gradients = extract_gradients(selection_model, D_train, D_val_preference)
influence_scores = compute_influence_scores(training_gradients, validation_gradients)
selected_data = select_top_data(D_train, influence_scores)
final_model = train_final_model(selected_data, final_model)
```

## Experiments

### List of Experiments

- Comparison with various baselines on different datasets (Table 1)
- Results of ROSE on different model architectures (Table 2)
- Validation Loss vs. Test Win Rate correlation analysis (Figure 3)
- Impact of the number of checkpoints on data selection (Table 3)
- Performance comparison across different validation shots (Figure 4)
- Transfer ability analysis across datasets and models (Table 9)
- Subtask results in benchmark datasets (Tables 10, 11, 12)

## Proofs

### List of Proofs

1. **Submodularity of Influence Function**: Submodularity in the influence function is analyzed via gradient-based estimation techniques, used in the ROSE method, showcasing submodularity in data selection.

2. **Lower Bound Termination Guarantee**: The paper outlines ROSE's data selection optimization framework, guaranteeing process termination by setting lower bounds based on the reward function derived from reinforcement learning techniques.

3. **Time Complexity Analysis**: ROSE's influence computation reduces computational costs via projections (e.g., TRAK) and efficiency techniques (e.g., LoRA). Its feasibility, even for large datasets, is demonstrated with empirical efficiency comparisons in the experimental section.