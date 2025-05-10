# Harnessing Diversity for Important Data Selection in Pretraining Large Language Models

## Meta

* **Name**: Harnessing Diversity for Important Data Selection in Pretraining Large Language Models
* **Journal**: ICLR (International Conference on Learning Representations)
* **Year**: 2025
* **Authors**: Beijing Institute of Technology, SenseTime Research, Shanghai Artificial Intelligence Laboratory, University of Arizona, Renmin University of China
* **Code**: [Github link](https://anonymous.4open.science/r/Quad/)
* **One-liner**: Introduces Quad, a data selection method balancing quality and diversity to enhance pretraining of large language models.
* **Model**: GPT-4, LLaMA-3.1
* **Datasets**: SlimPajama, FineWeb, LAMBADA, Openwebmath, FLAN
* **Baselines**: Random sampling, Qurating, DSIR, PPL, MATES

## Formulas

Below is a detailed breakdown of each formula along with explanations for every variable appearing in them.

### Influence Function Equation

The influence function is given by:

\[
I_\theta(D_r, z) = -\nabla L(\theta, D_r) \, (H + \lambda I)^{-1} \, \nabla L(\theta, z)
\]

**Explanation**:

- \(\theta\): Vector of model parameters.
- \(D_r\): Reference dataset used for baseline gradient calculation.
- \(z\): Single data instance whose influence is evaluated.
- \(L(\theta, \cdot)\): Loss function measuring model error \(M\) parameterized by \(\theta\).
- \(\nabla L(\theta, D_r)\): Gradient of the loss with respect to \(\theta\) on \(D_r\).
- \(\nabla L(\theta, z)\): Gradient of the loss function on data instance \(z\).
- \(H\): Hessian matrix of the second-order partial derivatives of the loss.
- \(\lambda\): Regularization parameter for ensuring invertibility.
- \(I\): Identity matrix.
- \((H + \lambda I)^{-1}\): Regularized inverse Hessian.

In summary, \(I_\theta(D_r, z)\) estimates an individual training instance's \(z\) impact on model parameters if perturbed or removed, moderated by a reference dataset \(D_r\) and a regularized curvature approximation.

### Cluster Score Equation

The cluster score is formulated as:

\[
CS_i = \bar{I}_i + \alpha \sqrt{\frac{2\ln\sum_j T(C_j)}{T(C_i)}}
\]

**Explanation**:

- \(CS_i\): Score assigned to cluster \(i\), balancing quality and diversity.
- \(\bar{I}_i\): Average influence score of instances in cluster \(i\).
- \(\alpha\): Tuning parameter weighing exploration against average influence.
- \(T(C_i)\): Frequency count for cluster \(C_i\).
- \(\sum_j T(C_j)\): Total number of samples across all clusters.
- \(\ln\): Natural logarithm.

This term acts as an upper confidence bound (UCB), enhancing scores for exploration-prone, less-sampled clusters while considering average influence.

### Influence Function with Attention Layers

The extension of the influence function to multi-head attention layers involves:

#### Influence in Attention

\[
I_{\theta_{att}}(D_r, z) = I_{\theta_{qkv}}(D_r, z) + I_{\theta_o}(D_r, z)
\]

**Explanation**:

- Influence decomposed into contributions from Query-Key-Value (QKV) and output projections.

#### Attention Forward Propagation

\[
Attention(Q, K, V) = \softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

**Explanation**:

- \(Q\): Query matrix; \(K\): Key matrix; \(V\): Value matrix.
- Division by \(\sqrt{d_k}\) maintains stable gradients.

### Training Flow

**Objective**: Select a high-quality and diverse subset \( D_b \) from \( D_c \) for pre-training.

1. **Clustering**: Organize \( D_c \) into clusters that are internally similar but externally diverse.
2. **Score Calculation**: Estimate average influence scores for clusters using attention-based Transformer architecture.
3. **Multi-Armed Bandit Framework**:
   - Treat clusters as arms.
   - Select top-k clusters with highest UCB scores.
   - Compute influence of selected cluster data incorporating attention layers.
4. **Iterative Update**: Ensure quality-diversity balance by updating cluster scores and sampling both high-quality and less-sampled clusters.

### Pseudocode

```python
def quad_data_selection(candidate_pool, reference_set, model):
    clusters = cluster_data(candidate_pool)
    cluster_scores = initialize_cluster_scores(clusters)
    selected_data = []

    while training:
        top_clusters = select_top_k_clusters(clusters, cluster_scores)
        for cluster in top_clusters:
            sampled_data = sample_data(cluster)
            influences = compute_influences(sampled_data, reference_set, model)
            update_cluster_score(cluster, influences)

        selected_data += select_influential_data(influences, threshold)
    
    return selected_data    
```

**Explanation**:
- **Clustering**: Organize data into clusters.
- **MAB**: Select clusters considering influence and diversity.
- **Influence Calculation**: Efficient influence calculation using Kronecker products.

## Inference Flow

### Inference Flow

1. **Data Clustering**: Organize \( D_c \) into clusters.
2. **Multi-Armed Bandit Framework**:
   - Each cluster as an arm.
   - Compute Cluster Score \( CSi \).
   - Select top-K clusters.
3. **Influence Calculation**:
   - Sample data to compute influence.
   - Use Kronecker product for Hessian computation.
4. **Data Sampling and Selection**:
   - Select samples with updated influence for decision making.
   - Ensure diversity by exploring less-frequently sampled clusters.
5. **Model Training**:
   - Incorporate selected data with high influence into the training set.

### Inference Flow Code

```python
import torch
from bandit import MultiArmedBandit

clusters = cluster_data(D_c) 
mab = MultiArmedBandit()

for iteration in range(num_iterations):
    selected_clusters = mab.select_top_clusters(clusters)
    
    for cluster in selected_clusters:
        data_samples = sample_cluster(cluster)
        influences = compute_influence(data_samples, reference_set)
        cluster.update_influence(influences)
    
    for cluster in clusters:
        if cluster.influence_above_threshold():
            add_to_training_set(cluster.select_data())

train_llm(training_data)
```

The pseudocode outlines the strategic sampling and influence computation tailored to attention architectures, ensuring quality maintenance with diversity for model training.

## Experiments

### List of Experiments

* Annotation budget ablations (Table 5)
* Time cost comparison between different sampling thresholds (Table 11)
* Evaluation of influence calculation accuracy on MLP vs attention layers (Figure 4b)
* Candidate score vs performance on downstream task correlation (Figure 4c)
* Ablation study on influence threshold \(\tau\) (Figure 4d)
* Exploration of MAB vs top-k clusters (Figure 4a)
* Continuous pretrain exploration with Openwebmath as reference set (Figure 7d)
* Clustering algorithm effect (Figure 5c)
* Impact of number of clusters (Figure 5d)
* Efficiency comparison of Quad and direct training with clean data (Figure 7c)
* Min-max-mean results across seeds and statistical significance (Table 6, 7)

## Proofs

### List of Proofs

* **Submodularity of Influence Function**: Demonstrates submodular properties crucial for selecting diverse influencing data points efficiently.
* **Lower Bound Termination Guarantee**: Ensures the data selection protocol terminates, balancing data quality and diversity to meet minimum performance.
* **Time Complexity Analysis**: Demonstrates efficiency in influence score computation using Hessian approximation techniques, scalable to large datasets.

This harmonized markdown provides a structured overview of the paper's key concepts, methods, and contributions, facilitating a comprehensive understanding of the proposed data selection approach for pretraining large language models.
