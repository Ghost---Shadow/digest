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

Here is a detailed breakdown of each variable in the key formulas, along with explanations of their roles.

### Influence Function

The influence of a data instance \(z\) on a model \(\theta\) is computed as:

\[ I_\theta(D_r, z) = -\nabla L(\theta, D_r) \, (H + \lambda I)^{-1} \, \nabla L(\theta, z) \]

**Variable Breakdown**:
- **\(I_\theta(D_r, z)\)**: Influence of data instance \(z\) on model parameters \(\theta\), relative to a reference dataset \(D_r\).
- **\(\nabla L(\theta, D_r)\)**: Gradient of the loss function \(L\) with respect to \(\theta\), calculated over the dataset \(D_r\).
- **\(\nabla L(\theta, z)\)**: Gradient of the loss function \(L\) with respect to \(\theta\) for a single data instance \(z\).
- **\(H\)**: Hessian matrix, second derivative of the loss \(L\) with respect to \(\theta\).
- **\(\lambda\)**: Regularization parameter ensuring \(H + \lambda I\) is invertible.
- **\(I\)**: Identity matrix.

### Cluster Score (CS)

Using an Upper Confidence Bound (UCB) method:

\[ CS_i = \bar{I}_i + \alpha \sqrt{\frac{2 \ln \left(\sum_{j} T(C_j)\right)}{T(C_i)}} \]

**Variable Breakdown**:
- **\(CS_i\)**: Score for cluster \(C_i\).
- **\(\bar{I}_i\)**: Average influence score of instances within cluster \(C_i\).
- **\(T(C_i)\)**: Number of times data instances have been sampled from cluster \(C_i\).
- **\(\sum_{j} T(C_j)\)**: Total samples across all clusters.
- **\(\alpha\)**: Hyperparameter balancing exploration and exploitation.

### Kronecker Product for Attention Layers

The Hessian matrix for QKV layers is approximated as:

\[ H_{qkv} = E\left(\delta_{qkv}\delta_{qkv}^T\right) \otimes E\left(x_{qkv}x_{qkv}^T\right) \]

**Variable Breakdown**:
- **\(H_{qkv}\)**: Approximated Hessian for QKV layers.
- **\(\delta_{qkv}\)**: Gradient information for QKV layers.
- **\(x_{qkv}\)**: Input vector for QKV layers.
- **\(E\left(\delta_{qkv}\delta_{qkv}^T\right)\)**: Expectation of outer product of \(\delta_{qkv}\).
- **\(E\left(x_{qkv}x_{qkv}^T\right)\)**: Expectation of outer product of \(x_{qkv}\).
- **\(\otimes\)**: Kronecker product operator.

These formulas showcase how quality and diversity are balanced in the Quad method, improving model performance by selecting influential and diverse data for pretraining.

## Training Flow

1. **Define Problem**: Select subset \(D_b\) from a large candidate pool \(D_c\) for pre-training with respect to reference \(D_r\).

2. **Quad Method Application**:
   - **Clustering**: Cluster \(D_c\) into groups, ensuring similarity within and diversity across groups.
   - **Score Calculation**: Estimate average influence score using an attention-based architecture.

3. **Data Quality and Diversity Balance**:
   - **Multi-Armed Bandit (MAB) Framework**:
     - Treat each cluster as an arm in MAB.
     - Select top-k clusters based on UCB scores.

4. **Iterative Update**:
   - Update cluster scores and sample data, considering both high-quality and less-sampled clusters.

5. **Threshold Selection**: Add data with influence scores above a defined threshold to the training dataset.

### Training Flow Code (High-level Pseudocode)

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

## Inference Flow

1. **Data Clustering**: Organize \(D_c\) into clusters using K-means.
2. **Multi-Armed Bandit Framework**:
   - Treat clusters as arms and compute \(CS_i\).
   - Select top-K clusters for further processing.
3. **Influence Calculation**:
   - Sample data to compute influence scores using the influence function.
4. **Data Sampling and Selection**:
   - Update sampling priorities and refine selection.
5. **Model Training**:
   - Incorporate selected data into training to minimize loss on \(D_r\).

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

## Experiments

### Experiment List

- Annotation budget ablations (Table 5)
- Time cost comparison between different sampling thresholds (Table 11)
- Evaluation of influence calculation accuracy on MLP vs. attention layers (Figure 4b)
- Candidate score vs. performance on downstream task correlation (Figure 4c)
- Ablation study on influence threshold \(\tau\) (Figure 4d)
- Exploration of MAB vs. top-k clusters (Figure 4a)
- Continuous pretrain exploration with Openwebmath as a reference set (Figure 7d)
- Clustering algorithm effect (Figure 5c)
- Impact of the number of clusters (Figure 5d)
- Efficiency comparison of Quad and direct training with clean data (Figure 7c)
- Min-max mean results across seeds and statistical significance (Table 6, 7)

## Proofs

### List of Proofs

1. **Submodularity of Influence Function**:
   - Demonstrates the influence function's submodular properties, essential for efficiently selecting influential data points.

2. **Lower Bound Termination Guarantee**:
   - Provides theoretical analysis ensuring that the algorithm meets a minimum performance threshold, balancing quality and diversity.

3. **Time Complexity Analysis**:
   - Examines the efficiency of the data selection method, particularly in computing influence scores using Kronecker product techniques.

This markdown document provides a comprehensive and harmonious summary of the paper, detailing the Quad method's innovative approach to balancing quality and diversity in data selection for pretraining large language models.