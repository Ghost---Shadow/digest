# ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning

## Meta

- **Title**: ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning
- **Journal**: Under review at the International Conference on Learning Representations (ICLR) 2025
- **Year**: 2025
- **Author**: Anonymous (under double-blind review)
- **Code**: To be made publicly available upon acceptance
- **One Liner**: How much does the validation score improve if we do a step on a particular data sample? Use a proxy model to estimate that, that is quality.
- **Models**: Llama (LLAMA-2-7B, LLAMA-2-13B, LLAMA-3.1-8B, LLAMA-3.1-8B-INS), Mistral (MISTRAL-7B-V0.3, MISTRAL-7B-INS-V0.3)
- **Datasets**: DOLLY, OPEN ASSISTANT 1, FLAN V2, COT; Evaluation: SHP, Stack Exchange (SE), HH-RLHF
- **Baselines**: Random, BM25, Representation-based data selection (RDS), DSIR, Influence Functions, LESS, Shapley values

## Formulas

### Cross-Entropy Loss for Instruction Tuning

The cross-entropy loss is defined as:
$$
\text{Cross-Entropy Loss} = -\sum \text{target} \; \log(\text{prediction})
$$

#### Components Breakdown

- **Target**: Represents the true token or one-hot encoded vector indicating the correct token the model is expected to predict.
- **Prediction**: The model’s predicted probability for the correct token, typically after a softmax layer.
- **Summation ($\sum$)**: Taken over all tokens in the sequence.

### Influence Estimation Scheme

Influence estimation is captured by:
$$
L(D_{\text{val}}; \theta_t) - L(D_{\text{val}}; \theta_{t-1}) = \langle \nabla_\theta L(D_{\text{val}}; \theta_{t-1}),\, \delta \theta \rangle
$$

#### Variables Breakdown

- **$L(D_{\text{val}}; \theta)$**: Loss evaluated on validation dataset.
- **$\theta_{t-1}$, $\theta_{t}$**: Model parameters before/after update.
- **Gradient terms** and **Delta parameter ($\delta \theta$)**: Capture parameter change effects.

### Reward-Oriented Optimization Framework

The reward function is:
$$
r(x, y) = \beta \log \!\left( \frac{\Omega_\theta(y \mid x)}{\Omega_{\text{ref}}(y \mid x)} \right) + \beta \log Z(x)
$$

And the pairwise preference optimization loss is:
$$
L_{\text{ROSE}}(\Omega_\theta; \Omega_{\text{ref}}) = -\mathbb{E}_{(x,y^w,y^l)\sim D'_{\text{val}}} \!\left[ \log \sigma\!\left( \beta \log \frac{\Omega_\theta(y^w \mid x)}{\Omega_{\text{ref}}(y^w \mid x)} - \beta \log \frac{\Omega_\theta(y^l \mid x)}{\Omega_{\text{ref}}(y^l \mid x)} \right) \right]
$$

#### Variables Breakdown

- **$r(x, y)$**: Reward to an output to compare preferences.
- **$L_{\text{ROSE}}$**: Loss function for pairwise preferences.

### Gradient for Optimizing ROSE Loss

The gradient is:
$$
\nabla_\theta L_{\text{ROSE}}(\Omega_\theta; \Omega_{\text{ref}}) = - \beta \, \mathbb{E}_{(x, y^w, y^l)\sim D'_{\text{val}}} \!\left[ \sigma\left(\hat{r}_\theta(x, y^l) - \hat{r}_\theta(x, y^w)\right) \cdot \left( \nabla_\theta \log \Omega_\theta(y^w \mid x) - \nabla_\theta \log \Omega_\theta(y^l \mid x) \right) \right]
$$

## Training Flow

1. **Model Initialization**
   - Start with models $\Omega$ and $\Gamma$, initializing with a 5% random subset of training data.
   
2. **Preference Validation Set Transformation**
   - Transform validation data to preference format.
  
3. **Gradient Calculation**
   - Compute gradients using models with new parameters and specific methods.
   
4. **Influence Score Evaluation**
   - Utilize gradient loss data to estimate influence scores.
  
5. **Data Selection**
   - Select top 5% of data points using the highest scores for training.
  
6. **Final Model Training**
   - Train $\Gamma$ with selected data, optimizing task-specific instruction.
  
7. **Output Fine-Tuned Model**
   - The optimized final model $\Gamma'$.

### Training Flow Code

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

1. **Transform Validation:**
   - Transition validation set to preference set.
   
2. **Backpropagation:**
   - Extract gradients of the training and transformed validation data.
  
3. **Influence Scoring:**
   - Use gradients to compute influence scores.
   
4. **Data Selection:**
   - Identify the top 5% influential data points.
  
5. **Final Training:**
   - Optimize the final model using selected data.

### Inference Flow Code

```python
# Transform validation set to preference set
def transform_to_preference_set(D_val):
    D_val_preference = [] 
    for prompt, response in D_val: 
        winning_response, losing_response = evaluate_responses(prompt, response) 
        D_val_preference.append((prompt, winning_response, losing_response)) 
    return D_val_preference

# Backpropagate and extract gradients
def extract_gradients(model, D, D_val_preference):
    training_gradients = []
    validation_gradients = []
    for data_point in D:
        gradient = model.compute_gradient(data_point)
        training_gradients.append(gradient)
    for prompt, win_res, loss_res in D_val_preference:
        gradient = model.compute_preference_gradient(prompt, win_res, loss_res)
        validation_gradients.append(gradient)
    return training_gradients, validation_gradients

# Compute influence scores
def compute_influence_scores(training_gradients, validation_gradients):
    influence_scores = []
    for train_grad in training_gradients:
        score = sum(np.dot(val_grad, train_grad) for val_grad in validation_gradients)
        influence_scores.append(score)
    return influence_scores

# Select the top data points
def select_top_data(D, influence_scores, top_percentage=0.05):
    num_top_points = int(len(D) * top_percentage)
    top_indices = np.argsort(influence_scores)[-num_top_points:]
    return [D[i] for i in top_indices]

# Final model training
def train_final_model(selected_data, model):
    model.train_on(selected_data)
    return model

D_val_preference = transform_to_preference_set(D_val)
training_gradients, validation_gradients = extract_gradients(selection_model, D_train, D_val_preference)
influence_scores = compute_influence_scores(training_gradients, validation_gradients)
selected_data = select_top_data(D_train, influence_scores)
final_model = train_final_model(selected_data, final_model)
```

## Experiments

### List of Experiments

- **Baseline Comparison** on datasets (Table 1)
- **ROSE Performance** across models (Table 2)
- **Validation Loss vs. Test Win Rate** correlation (Figure 3)
- **Checkpoints Impact** on selection (Table 3)
- **Validation Shots Performance** (Figure 4)
- **Transfer Ability Analysis** on datasets/models (Table 9)
- **Subtask Results** for benchmark datasets (Tables 10, 11, 12)

## Proofs

### List of Proofs

1. **Submodularity of Influence Function**:  
   - Demonstrates influence estimation's submodular properties, enhancing data selection.
   
2. **Lower Bound Termination Guarantee**:  
   - Proves that data selection terminates with a bounded reward increase.
   
3. **Time Complexity Analysis**:  
   - Confirms ROSE's efficiency in influence computation and data scalability.