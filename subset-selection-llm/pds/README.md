# DATA SELECTION VIA OPTIMAL CONTROL FOR LANGUAGE MODELS

## Meta

* **Name**: DATA SELECTION VIA OPTIMAL CONTROL FOR LANGUAGE MODELS
* **Journal**: ICLR
* **Year**: 2025
* **Authors**: The CoAI Group, Tsinghua University; Microsoft Research; Peking University
* **Code**: [Github Link](https://github.com/microsoft/LMOps/tree/main/data_selection)
* **One-liner**: Utilizing Optimal Control theory to enhance language models by selecting high-quality pre-training data efficiently.
* **Model**: LIMA
* **Datasets**: CommonCrawl, LIMA
* **Baselines**: Conventional Pre-Training, RHO-Loss, DSIR, IF-Score

## Formulas and Key Concepts

### 1. Dynamics of Model Training

The evolution of model parameters is governed by the formula:

\[
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t, \gamma).
\]

- \(\theta_t\), \(\theta_{t+1}\): Model parameters at step \(t\) and \(t+1\).
- \(\eta\): Learning rate.
- \(\nabla L(\theta_t, \gamma)\): Gradient of the loss function.
- \(L(\theta, \gamma)\): Defined as \(\sum_{n=1}^{|D|} \gamma_n\, l(x_n, \theta)\).

### 2. Overall Optimization Problem

The goal is to:

\[
\min_{\gamma} \sum_{t=1}^{T} J(\theta_t),
\]

subject to:

\[
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t, \gamma), \quad \gamma \in U.
\]

- \(J(\theta_t)\): Downstream loss function.
- \(\gamma\): Vector of data quality scores with constraints formed by simplex \(U\).

### 3. Pontryagin's Maximum Principle (PMP) Conditions

Key conditions include:

a) **State Evolution**:  
\[
\theta^*_{t+1} = \theta^*_t - \eta \nabla L(\theta^*_t, \gamma^*),
\]

b) **Co-state Dynamics**:  
\[
\lambda^*_t = \lambda^*_{t+1} + \nabla J(\theta^*_t) - \eta \nabla^2 L(\theta^*_t, \gamma^*) \lambda^*_{t+1},
\]

c) **Optimal Control Condition**:  
\[
\gamma^* = \arg \max_{\gamma} \left\{ \sum_{n=1}^{|D|} \gamma_n \sum_{t=0}^{T-1} \lambda^*_{t+1} \cdot \nabla l(x_n, \theta^*_t) \right\}.
\]

### 4. Implementation Considerations

- **PMP-based Data Selection (PDS) Framework**: Employing small proxy models and data scorers to infer and apply data selection efficiently.

## Training Flow

### Overview

1. Formulate the training problem as an optimal control problem.
2. Optimize data quality scores \(\gamma\) using Pontryagin's Maximum Principle.
3. Train with optimal conditions using proxy models for efficiency.
4. Apply the trained data scorer to larger datasets for pre-training.

### Pseudocode

```python
# Initialize
theta_proxy = initialize_proxy_model(N_proxy)
scorer_model = initialize_scorer_model(N_scorer)

# Compute quality scores
gamma = initialize_gamma(D_proxy)
for epoch in range(num_outer_epochs):
    for t in range(T_proxy):
        theta_proxy = gradient_update(theta_proxy, gamma, D_proxy)
    lambda = compute_lambda(theta_proxy, J, T_proxy)
    for n, x in enumerate(D_proxy):
        gamma[n] += alpha * sum([np.dot(lambda[t], grad_loss(x, theta_proxy[t])) for t in range(T_proxy)])
    gamma = project_to_simplex(gamma)

# Train data scorer and apply to full dataset
scorer_model = train_scorer(scorer_model, D_proxy, gamma)
scores = infer_scores(scorer_model, D)
selected_data = select_top_k(scores, D, ratio=r)

# Pre-train target model
theta_target = initialize_target_model(N)
trained_model = train_model(theta_target, selected_data, T_target)
```

## Inference Flow

### Overview

1. Initialize and train a proxy LM using PMP-derived conditions.
2. Compute co-state vectors and update scores \(\gamma\).
3. Train a data scorer on derived scores, and infer scores for complete data.
4. Perform Gumbel-Top-K sampling to select data for final training.

### Pseudocode

```python
def PMP_DataSelection(Dprx, LM_proxy, ..., learning_rate):
    γ = torch.full((len(Dprx),), 1/len(Dprx))
    for _ in range(num_iterations):
        θ, training_dynamics = LM_proxy.train(Dprx, γ, T)
        λ = compute_co_state_vectors(θ, training_dynamics, T)
        gradient_alignment = compute_gradient_alignment(λ, Dprx, θ)
        γ += learning_rate * gradient_alignment
        γ = project_to_simplex(γ)
    return γ

def TrainDataScorer(Dprx, γ, data_scorer_model):
    return data_scorer_model.train(Dprx, γ)

def InferAndSelect(D, ..., gumbel_strength):
    scores = data_scorer_model.infer(D)
    return gumbel_top_k(scores, selection_ratio, gumbel_strength)
```

## Experiments

1. Scaling with model size and computation (Figure 1).
2. Downstream evaluation performance (Tables 1 and 2, Figures 8).
3. Test losses on the DCLM corpus (Figure 4, Table 3).
4. Data utilization and reduction experiment (Figure 5).
5. Efficient data quality score implementation (Figure 6).
6. Loss scaling and extrapolation tests (Section I.4, Table 3, Figure 9).
7. Ablation studies on various parameters (Figures 10 and 11, Tables 5 and 10).

## Proofs

1. **Theorem 2.1**: PMP Conditions for Data Selection.
2. **Proof of Theorem 2.1**.
3. **Theorem C.1**: PMP Data Selection for Adam.

This markdown file aims to ensure consistent formatting and presentation of the paper content, providing a clear and concise overview of the approach, its components, and associated experimental results.