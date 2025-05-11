# Task-Adaptive Pretrained Language Models via Clustered Importance Sampling

## Meta

- **Name:** Task-Adaptive Pretrained Language Models via Clustered Importance Sampling
- **Journal:** International Conference on Learning Representations (ICLR)
- **Year:** 2025
- **Author:** Apple
- **Code:** Not available
- **One-liner:** The paper proposes a novel method, CRISP, for building task-specialist language models by adapting training data from large generalist datasets through clustered importance sampling.
- **Model:** Not specified; implied models range from 350M to 7B parameter LLMs.
- **Datasets:** Redpj2, PubMed Central, StackExchange, Wikipedia, AI2 Reasoning Challenge (ARC), Massive Multitask Language Understanding (MMLU), and Reward Bench Reasoning (RWDB-R).
- **Baselines:** Fine-tuning generalist models, task-specific pretraining, DoGE method, cross-entropy difference (CED) method.

## Formulas

### 1. General Loss Function

The loss of a model with parameters $\theta$ on a dataset $D$ is defined as:

$$
L(D; \theta) := \frac{1}{|D|}\sum_{x \in D} \ell(x; \theta) = - \frac{1}{|D|}\sum_{x \in D} \frac{1}{|x|} \sum_i \log p(x_i \mid x_{1}^{i-1}; \theta)
$$

- $\theta$: Model parameters.
- $D$: Dataset.
- $|D|$: Number of sequences in $D$.
- $x$: One sequence in $D$.
- $|x|$: Number of tokens in $x$.
- $x = (x_1, x_2, \ldots, x_{|x|})$: Tokenized sequence.
- $\ell(x; \theta)$: Loss for sequence $x$ given $\theta$.
- $p(x_i \mid x_{1}^{i-1}; \theta)$: Probability of $i$-th token given previous tokens as predicted by the model.

### 2. Perplexity

The perplexity of the model $\theta$ on dataset $D$ is:

$$
P(D; \theta) := \exp(L(D; \theta))
$$

- $P(D; \theta)$: Model perplexity.
- $\exp(L(D; \theta))$: Exponential of the loss.

### 3. Importance Weights for Clusters

CRISP modifies the training distribution by assigning importance weights to clusters:

$$
w(c) = \frac{P(c\mid D_s)}{P(c\mid D_g)}
$$

- $w(c)$: Importance weight for cluster $c$.
- $P(c \mid D_s)$: Probability of $c$ under the specialist distribution.
- $P(c \mid D_g)$: Probability of $c$ under the generalist distribution.

### 4. Independence Assumption

Assumption simplifies expressions:

$$
P(x\mid c, D_s) = P(x\mid c)
$$

- $P(x\mid c, D_s)$: Probability of sample $x$ given $c$ and dataset $D_s$.
- Assumes $P(x\mid c)$ independent of the dataset.

### 5. Loss Computation by Cluster

Specialist dataset loss:

$$
L(D_s; \theta) = \sum_c L(c; \theta) P(c\mid D_s)
$$

Generalist dataset loss:

$$
L(D_g; \theta) = \sum_c L(c; \theta) P(c\mid D_g)
$$

Cluster-specific loss:

$$
L(c; \theta) = \sum_x \ell(x; \theta) P(x\mid c)
$$

## Training Flow

### 1. Dataset and Clustering Initialization

- Start with generalist dataset $D_g$ and specialist dataset $D_s$.
- Cluster $D_g$ and calculate cluster histogram based on $D_s$.

### 2. Model Initialization

- Initialize language model with initial parameters $\theta_0$.

### 3. Importance Sampling Weight Calculation

- Compute importance weight for each cluster $c$ as $w(c) = \frac{P(c | D_s)}{P(c | D_g)}$.

### 4. Training Process

- For each training step:
  1. Sample a cluster id $c$.
  2. Retrieve examples $x$ from $D_g$ of cluster $c$.
  3. Update model parameters $\theta_t$ by minimizing weighted loss using stochastic gradient descent.

### 5. Loss Calculation

- Use the loss function:

  $$
  L(D; \theta) = -\frac{1}{|D|} \sum_{x \in D} \frac{1}{|x|} \sum_{i} \log p(x_i|x_{i-1}^1; \theta)
  $$

### 6. Continuous Pretraining and Evaluation

- Perform continued pretraining using CRISP by splitting training into generic and task-dependent phases, iterating until convergence.
- Periodically evaluate performance on validation sets.

### Training Flow Code

```python
# Pytorch style pseudocode
initialize_clustering_histogram(D_g, D_s)
initialize_model()
for t in range(num_steps):
    sample_cluster = sample_categorical(clustering_histogram)
    retrieve_examples = sample_data(D_g, sample_cluster)
    loss = compute_loss(retrieve_examples, model)
    optimize_model(model, loss)
    evaluate_model(model, validation_data)
```

## Inference Flow

### 1. Calculate SBERT Embeddings

- Calculate SBERT embeddings for each token window of 1,024 tokens from $D_g$.

### 2. Cluster Generalist Dataset

- Cluster $D_g$ using hierarchical clustering based on SBERT embeddings.

### 3. Compute Cluster Histogram

- Compute $P(c|D_s)$ for specialist dataset $D_s$.

### 4. Estimate Importance Weights

- Estimate importance weights $w(c) = \frac{P(c|D_s)}{P(c|D_g)}$.

### 5. Sample Clusters

- Sample clusters from $D_g$ based on $w(c)$ to form a new training dataset.

### 6. Train Language Model

- Train a specialist language model using sampled data.

### 7. Evaluate Language Model

- Evaluate performance using perplexity or accuracy on task-dependent tasks.

### Inference Flow Code

```python
import torch

sbert_model = load_sbert_model()
hierarchical_clusters = cluster_data(generalist_data, sbert_model)

def calculate_importance_weights(specialist_data, generalist_data, hierarchical_clusters):
    cluster_specialist_hist = count_clusters(specialist_data, hierarchical_clusters)
    cluster_generalist_hist = count_clusters(generalist_data, hierarchical_clusters)
    importance_weights = {c: cluster_specialist_hist[c] / cluster_generalist_hist[c] for c in hierarchical_clusters}
    return importance_weights

importance_weights = calculate_importance_weights(specialist_data, generalist_data, hierarchical_clusters)

def sample_clusters(importance_weights, generalist_data):
    sampled_data = []
    for cluster, weight in importance_weights.items():
        sampled_tokens = sample_from_cluster(cluster, generalist_data, weight)
        sampled_data.extend(sampled_tokens)
    return sampled_data

model = LanguageModel()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for batch in data_loader(sample_clusters(importance_weights, generalist_data)):
        predictions = model(batch)
        loss = compute_loss(predictions, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    validate_model(model)
```

## Experiments

### List of Experiments

- Annotation budget ablations on generalist and specialist datasets.
- Evaluation of pretraining vs. fine-tuning on language modeling tasks across data scales.
- Evaluation of pretraining vs. fine-tuning on multiple choice questions across datasets.
- Comparison of importance sampling with cross-entropy difference and other methods.
- Task-transfer and multitasking evaluation for different model sizes.
- Impact of clustering representation and number of clusters on performance.
- Performance analysis of models with varying sizes and training costs.
- Impact of different amounts of training data on model performance.

## Proofs

### Theoretical Concepts and Evaluations

- **Importance Sampling Application**: Discusses the theory and mathematical framework for using importance sampling.
- **Comparison of Weighting Methods**: Empirical comparison of DoGE and CRISP weights over generalist and specialist clusters.

The paper frames these as methodological approaches with empirical backing, rather than formal proofs.