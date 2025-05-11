# Data Selection via Optimal Control for Language Models

## Meta

* **Name**: DATA SELECTION VIA OPTIMAL CONTROL FOR LANGUAGE MODELS
* **Journal**: ICLR
* **Year**: 2025
* **Authors**: 
  1. The CoAI Group, Tsinghua University 
  2. Microsoft Research 
  3. Peking University
* **Code**: [GitHub Repository](https://github.com/microsoft/LMOps/tree/main/data_selection)
* **One-liner**: Using Optimal Control theory to select high-quality pre-training data for improving language models' efficiency and performance.
* **Model**: LIMA
* **Datasets**: CommonCrawl, LIMA
* **Baselines**: Conventional Pre-Training, RHO-Loss, DSIR, IF-Score

## Formulas

### General Pre-training Loss

The pre-training loss is defined as:

$$
L(\theta, \gamma) = \sum_{n=1}^{|D|} \gamma_n\, l(x_n, \theta),
$$

where:

- $l(x_n, \theta) = -\log p_\theta(x_n)$, with:
  - $\theta$: Model parameters.
  - $\gamma$: Data selection weights.
  - $x_n$: Instance from dataset $D$.
  - $|D|$: Total instances in $D$.
  - $p_\theta(x_n)$: Probability of $x_n$.

### Optimization Problem for Data Selection

The optimization problem is:

$$
\min_{\gamma} \sum_{t=1}^{T} J(\theta_t),
$$

subject to:

$$
\theta_{t+1} = \theta_t - \eta\, \nabla L(\theta_t, \gamma), \quad \gamma \in U.
$$

where:

- $J(\theta_t)$: Downstream loss at step $t$.
- $T$: Total training steps.
- $\eta$: Learning rate.
- $\nabla L(\theta_t, \gamma)$: Gradient of pre-training loss.

### Data Selection as Optimal Control

Data selection is interpreted as an optimal control problem:

#### State Transition Equation

$$
\theta_{t+1} = \theta_t - \eta\, \nabla L(\theta_t, \gamma), \quad \gamma \in U.
$$

#### Co-state Equation

$$
\lambda_t = \lambda_{t+1} + \nabla J(\theta_t) - \eta\, \nabla^2 L(\theta_t, \gamma)\, \lambda_{t+1}.
$$

#### Necessary Optimality Condition for Data Selection

$$
\gamma^* = \arg\max_{\gamma} \sum_{n=1}^{|D|} \gamma_n \sum_{t=0}^{T-1} \lambda_{t+1}^\top\, \nabla l(x_n, \theta_t).
$$

## Training Flow

### Training Flow Steps

1. Formulate pre-training as an optimal control problem.
2. Optimize data quality scores $\gamma$.
3. Solve for $\gamma$ using Pontryagin’s Maximum Principle.
4. Use proxy models for training dynamics.
5. Use a data scorer to predict scores.
6. Pre-train the target model on selected data.

### Training Flow Code

```python
# Initialize parameters for proxy model and scorer
theta_proxy = initialize_proxy_model(N_proxy)
scorer_model = initialize_scorer_model(N_scorer)

# Compute data quality scores on a subset D_proxy
gamma = initialize_gamma(D_proxy)
for epoch in range(num_outer_epochs):
    for t in range(T_proxy):
        theta_proxy = gradient_update(theta_proxy, gamma, D_proxy)

    lambda = compute_lambda(theta_proxy, J, T_proxy)

    for n, x in enumerate(D_proxy):
        gamma[n] += alpha * sum([np.dot(lambda[t], grad_loss(x, theta_proxy[t])) for t in range(T_proxy)])

    gamma = project_to_simplex(gamma)

scorer_model = train_scorer(scorer_model, D_proxy, gamma)

scores = infer_scores(scorer_model, D)
selected_data = select_top_k(scores, D, ratio=r)

theta_target = initialize_target_model(N)
trained_model = train_model(theta_target, selected_data, T_target)
```

## Inference Flow

### Inference Flow Steps

1. Initialize data quality scores γ.
2. Train a proxy LM using gradient descent.
3. Compute co-state vectors λ.
4. Update γ based on gradient alignment with λ.
5. Train a data scorer model on γ.
6. Use the data scorer to infer scores on the full corpus.
7. Use Gumbel-Top-K sampling to select data for the target LM.

### Inference Flow Code

```python
def PMP_DataSelection(Dprx, LM_proxy, num_iterations, learning_rate):
    γ = torch.full((len(Dprx),), 1/len(Dprx))
    
    for _ in range(num_iterations):
        θ, training_dynamics = LM_proxy.train(Dprx, γ, T)
        λ = compute_co_state_vectors(θ, training_dynamics, T)
        gradient_alignment = compute_gradient_alignment(λ, Dprx, θ)
        γ += learning_rate * gradient_alignment
        γ = project_to_simplex(γ)
    
    return γ

def TrainDataScorer(Dprx, γ, data_scorer_model):
    data_scorer_model.train(Dprx, γ)
    return data_scorer_model

def InferAndSelect(D, data_scorer_model, selection_ratio, gumbel_strength):
    scores = data_scorer_model.infer(D)
    selected_data = gumbel_top_k(scores, selection_ratio, gumbel_strength)
    return selected_data
```

## Experiments

### List of Experiments

1. Scaling of Training Computation and Model Size (Figure 1)
2. Downstream evaluation performance (Tables 1 and 2, Figures 8)
3. Test losses on DCLM corpus (Figure 4, Table 3)
4. Data utilization and reduction experiment (Figure 5)
5. Efficient data quality score implementation (Figure 6)
6. Scaling law test loss extrapolation (Section I.4, Table 3, Figure 9)
7. Ablation studies (Figures 10 and 11, Tables 5 and 10)

## Proofs

### List of Proofs

* **Theorem 2.1**: PMP Conditions for Data Selection
* **Proof of Theorem 2.1**
* **Theorem C.1**: PMP Data Selection for Adam