# Task-Adaptive Pretrained Language Models via Clustered Importance Sampling

## Meta

- **Name**: Task-Adaptive Pretrained Language Models via Clustered Importance Sampling
- **Journal**: International Conference on Learning Representations (ICLR)
- **Year**: 2025
- **Author**: Apple
- **Code**: Not available
- **One-liner**: The paper proposes a novel method, CRISP, for building task-specialist language models by adapting training data from large generalist datasets through clustered importance sampling.
- **Model**: Not specified; implied models range from 350M to 7B parameter Large Language Models (LLMs).
- **Datasets**: Redpj2, PubMed Central, StackExchange, Wikipedia, AI2 Reasoning Challenge (ARC), Massive Multitask Language Understanding (MMLU), and Reward Bench Reasoning (RWDB-R)
- **Baselines**: Fine-tuning generalist models, task-specific pretraining, DoGE method, cross-entropy difference (CED) method.

## Formulas

### General Loss Function

The loss of a model with parameters $\theta$ on a dataset $D$ is defined as:

$$
L(D; \theta) := \frac{1}{|D|}\sum_{x \in D} \ell(x; \theta) = - \frac{1}{|D|}\sum_{x \in D} \frac{1}{|x|} \sum_i \log p(x_i \mid x_{1}^{i-1}; \theta)
$$

- $\theta$: Parameters of the model.
- $D$: Dataset.
- $|D|$: Number of sequences in $D$.
- $x$: Sequence in the dataset $D$.
- $|x|$: Length of sequence $x$.
- $\ell(x; \theta)$: Loss for sequence $x$.
- $p(x_i \mid x_{1}^{i-1}; \theta)$: Probability of the $i$-th token given previous tokens.

### Perplexity

The perplexity of the model $\theta$ on dataset $D$ is given by:

$$
P(D; \theta) := \exp(L(D; \theta))
$$

- $P(D; \theta)$: Perplexity of the model.
- Lower perplexity indicates a better model.

### Importance Weights for Clusters

The importance weight for a cluster $c$ is defined as:

$$
w(c) = \frac{P(c\mid D_s)}{P(c\mid D_g)}
$$

- $w(c)$: Importance weight for cluster $c$.
- $P(c \mid D_s)$: Probability of cluster $c$ under the specialist distribution.
- $P(c \mid D_g)$: Probability of cluster $c$ under the generalist distribution.

### Independence Assumption

$$
P(x\mid c, D_s) = P(x\mid c)
$$

- $P(x \mid c, D_s)$: Probability of $x$ given cluster $c$ and $D_s$.
- States that once $c$ is known, $P(x \mid c)$ is independent of the dataset.

### Loss Computation by Cluster

For the specialist dataset $D_s$:

$$
L(D_s; \theta) = \sum_c L(c; \theta) P(c\mid D_s)
$$

For the generalist dataset $D_g$:

$$
L(D_g; \theta) = \sum_c L(c; \theta) P(c\mid D_g)
$$

- $L(c; \theta) = \sum_x \ell(x; \theta) P(x\mid c)$ is the loss associated with cluster $c$.

### Summary

The CRISP method:
- Computes a basic token-level loss.
- Defines a perplexity metric.
- Utilizes importance sampling via cluster-level weight adjustments.
- Employs an independence assumption for simplification.
- Computes losses by marginalizing over clusters for alignment with specialist dataset’s distribution.

## Training Flow

### Training Flow

1. **Dataset and Clustering Initialization**: 
   - Use a generalist dataset $D_g$ and a specialist dataset $D_s$.
  
2. **Model Initialization**:
   - Initiate the language model with parameters $\theta_0$.

3. **Importance Sampling Weight Calculation**: 
   - Compute importance weights for each cluster.

4. **Training Process**: 
   - Perform training using sampled clusters and update model parameters.

5. **Loss Calculation**: 
   - Use the defined loss function for minimization.

6. **Continuous Pretraining and Evaluation**: 
   - Iterative pretraining and validation for optimal configurations.

### Training Flow Code

```python
# PyTorch-style pseudocode
initialize clustering_histogram(D_g, D_s)
initialize_model()
for t in range(num_steps):
    sample_cluster = sample_categorical(clustering_histogram)
    retrieve_examples = sample_data(D_g, sample_cluster)
    loss = compute_loss(retrieve_examples, model)
    optimize_model(model, loss)
    evaluate_model(model, validation_data)
```

## Inference Flow

### Inference Flow

1. Calculate SBERT embeddings for token windows.
2. Cluster the generalist dataset.
3. Compute the cluster histogram for the specialist dataset.
4. Estimate importance weights for clusters.
5. Sample clusters based on importance to form a new training dataset.
6. Train a specialist language model.
7. Evaluate model performance.

### Inference flow Code

```python
import torch

sbert_model = load_sbert_model()
hierarchical_clusters = cluster_data(generalist_data, sbert_model)

def calculate_importance_weights(specialist_data, generalist_data, hierarchical_clusters):
    cluster_specialist_hist = count_clusters(specialist_data, hierarchical_clusters)
    cluster_generalist_hist = count_clusters(generalist_data, hierarchical_clusters)
    importance_weights = {c: cluster_specialist_hist[c] / cluster_generalist_hist[c] 
                          for c in hierarchical_clusters}
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

- Annotation budget ablations on datasets.
- Evaluation of pretraining vs fine-tuning on language modeling tasks.
- Evaluation on multiple choice tasks across datasets.
- Comparison of importance sampling with other methods.
- Task-transfer and multitasking evaluation.
- Impact of clustering representation and number of clusters.
- Performance analysis of models with varying sizes.
- Impact of training data amounts on performance.

## Proofs

### List of Proofs

This paper does not explicitly provide formal proofs but discusses:

- **Importance Sampling Application**: Discusses using importance sampling for resampling the dataset, providing a framework through equations.
  
- **Comparison of Weighting Methods**: Compares DoGE and CRISP weights, showing similar distributions for top clusters.

These concepts are framed as methodological approaches with empirical backing rather than formal proofs.