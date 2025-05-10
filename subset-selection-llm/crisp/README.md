# Task-Adaptive Pretrained Language Models via Clustered Importance Sampling

## Meta

* **Name**: Task-Adaptive Pretrained Language Models via Clustered Importance Sampling
* **Journal**: International Conference on Learning Representations (ICLR)
* **Year**: 2025
* **Author**: Apple
* **Code**: Not available
* **One-liner**: The paper proposes a novel method, CRISP, for building task-specialist language models by adapting training data from large generalist datasets through clustered importance sampling.
* **Model**: Not specified; implied models range from 350M to 7B parameter LLMs.
* **Datasets**: Redpj2, PubMed Central, StackExchange, Wikipedia, AI2 Reasoning Challenge (ARC), Massive Multitask Language Understanding (MMLU), and Reward Bench Reasoning (RWDB-R)
* **Baselines**: Fine-tuning generalist models, task-specific pretraining, DoGE method, cross-entropy difference (CED) method.

## Formulas

The following section provides a breakdown of the key formulas used in the paper, explained with variables and their significance, presented in MathJax-style LaTeX.

### 1. Loss on a Dataset

The paper defines the loss of a model parameterized by \(\theta\) on a dataset \(D\) as:

\[
L(D; \theta) := \frac{1}{|D|} \sum_{x \in D} \ell(x; \theta) 
= -\frac{1}{|D|} \sum_{x \in D} \frac{1}{|x|} \sum_i \log p(x_i \mid x_{1}^{i-1}; \theta)
\]

**Explanation of Symbols:**

- \(\theta\): Model parameters.
- \(D\): Dataset used for calculating the loss.
- \(|D|\): Number of examples in \(D\).
- \(x \in D\): A sequence from the dataset \(D\).
- \(\ell(x; \theta)\): Loss for a sequence \(x\).
- \(|x|\): Length of sequence \(x\).
- \(x = (x_1, x_2, \dots, x_{|x|})\): Ordered tokens in \(x\).
- \(\sum_i \log p(x_i\mid x_{1}^{i-1}; \theta)\): Log probability of tokens given previous tokens.

The formula describes the average token-level negative log-likelihood across the dataset.

### 2. Perplexity

The perplexity of the model is defined as:

\[
P(D; \theta) := \exp(L(D; \theta))
\]

- **Perplexity \(P(D; \theta)\)**: Measures how well the model predicts the test data. Lower perplexity indicates better performance.

### 3. Objective on the Specialist Distribution (CRISP)

The loss on a specialist distribution \(D_s\) is defined as:

\[
L(D_s; \theta) = E_{x \sim D_s}[\ell(x; \theta)] 
= \sum_x \ell(x; \theta) P(x \mid D_s)
\]

This is the expected loss computed over the specialist data distribution \(D_s\).

### 4. Incorporating the Cluster Variable \(c\)

The loss marginalized over a latent cluster variable \(c\) is:

\[
L(D_s; \theta) = \sum_c \sum_x \ell(x; \theta) P(x \mid c, D_s) P(c \mid D_s)
\]

Assuming independence:

\[
P(x \mid c, D_s) = P(x \mid c)
\]

The simplified loss:

\[
L(D_s; \theta) = \sum_c \sum_x \ell(x; \theta) P(x \mid c) P(c \mid D_s)
\]

### 5. Expressing Loss as an Expectation over Clusters

The loss becomes:

\[
L(D_s; \theta) = E_{c \sim P(c \mid D_s)}[L(c; \theta)]
\]

Where:

\[
L(c; \theta) = \sum_x \ell(x; \theta) P(x \mid c)
\]

### 6. Re-weighting with Importance Sampling

The loss reweighted via importance sampling:

\[
L(D_s; \theta) = \sum_c L(c; \theta) \frac{P(c \mid D_s)}{P(c \mid D_g)} P(c \mid D_g)
= E_{c \sim P(c \mid D_g)}\Bigl[w(c) L(c; \theta)\Bigr]
\]

**Importance Weights**: 

\[
w(c) = \frac{P(c \mid D_s)}{P(c \mid D_g)}
\]

This adjusts for differences in cluster prevalence between the specialist and generalist datasets.

## Training Flow

### Overview

1. **Dataset and Clustering Initialization**:
   - Initialize with datasets \(D_g\) (generalist) and \(D_s\) (specialist).
   - Cluster \(D_g\) and create a histogram based on \(D_s\).

2. **Model Initialization**:
   - Start with initial model parameters \(\theta_0\).

3. **Importance Sampling Weight Calculation**:
   - Calculate \(w(c) = \frac{P(c | D_s)}{P(c | D_g)}\) for each cluster.

4. **Training Process**:
   - Iterate through training steps, sampling clusters based on weights and updating model parameters \(\theta_t\).

5. **Loss Calculation**:
   - Employ the loss function:
     \[
     L(D; \theta) = -\frac{1}{|D|} \sum_{x \in D} \frac{1}{|x|} \sum_{i} \log p(x_i|x_{i-1}^1; \theta)
     \]

6. **Continuous Pretraining and Evaluation**:
   - Conduct pretraining using CRISP and evaluate on validation sets.

### Code Pseudocode

```python
# Pytorch style pseudocode
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

### Key Steps

1. Calculate SBERT embeddings for each token window from \(D_g\).
2. Cluster \(D_g\) using embeddings to form variously sized clusters.
3. Compute specialist dataset cluster histogram \(P(c|D_s)\).
4. Calculate importance weights \(w(c)\) and sample data accordingly.
5. Train the specialist language model with sampled data.
6. Evaluate performance on validation sets using appropriate metrics.

### Code Pseudocode

```python
import torch

# Assumed pre-loaded data
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

# Train language model with the sampled data
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

This pseudocode illustrates the CRISP method's key steps, from hierarchical clustering to training a model using importance sampling.

## Experiments

### List of Experiments

* **Annotation Budget Ablations**: Analyze effects of annotation budgets (Section 5.3, Figure 11).
* **Evaluation of Pretraining vs Fine-tuning**: Compare learning methods across scales and tasks (Section 4.1 and 4.2, Figures 2-4).
* **Comparison with Other Methods**: Evaluate weighting methods contrasting importance sampling with cross-entropy difference (Section H, Table 14).
* **Task and Multitasking**: Assess cross-task effects for different model sizes (Section 5.4, Tables 1 and 15).
* **Performance Impact Due to Clustering**: Evaluate how representation and cluster count affect outcomes (Section 5.1, Figures 5-8).
* **Performance and Training Cost Analysis**: Examine how model size and cost relate to performance (Section 5.2, Figures 9 and 10).
* **Impact of Training Data**: Investigate different training data volumes on performance (Section 5.3, Figure 11).

## Proofs

### List of Proofs

- **Importance Sampling Application**: Discusses the theoretical underpinning of importance sampling, presented through mathematical formulation rather than formal proofs.
- **Method Comparisons**: Empirical evaluations contrast CRISP with DoGE and other methods, focusing on application and results rather than detailed proofs.
