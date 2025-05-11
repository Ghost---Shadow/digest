# Harnessing Diversity for Important Data Selection in Pretraining Large Language Models

## Meta

- **Name**: Harnessing Diversity for Important Data Selection in Pretraining Large Language Models
- **Journal**: ICLR (International Conference on Learning Representations)
- **Year**: 2025
- **Authors**: Beijing Institute of Technology, SenseTime Research, Shanghai Artificial Intelligence Laboratory, University of Arizona, Renmin University of China
- **Code**: [GitHub link](https://anonymous.4open.science/r/Quad/)
- **One-liner**: Introduces Quad, a data selection method balancing quality and diversity to enhance pretraining of large language models.
- **Model**: GPT-4, LLaMA-3.1
- **Datasets**: SlimPajama, FineWeb, LAMBADA, Openwebmath, FLAN
- **Baselines**: Random sampling, Qurating, DSIR, PPL, MATES

## Key Formulas

### Influence Function

The influence of a data instance \( z \) on a model \( \theta \) is computed as:

\[
I_\theta(D_r, z) = -\nabla L(\theta, D_r) \, (H + \lambda I)^{-1} \, \nabla L(\theta, z)
\]

**Variables Breakdown:**

- **\( I_\theta(D_r, z) \)**: Influence of data instance \( z \) on model parameters \( \theta \) based on reference dataset \( D_r \).
- **\( \nabla L(\theta, D_r) \)**: Gradient of the loss function with respect to \( \theta \) over dataset \( D_r \).
- **\( \nabla L(\theta, z) \)**: Gradient of the loss function for a single instance \( z \).
- **\( H \)**: Hessian matrix (second derivative) of the loss function about \( \theta \).
- **\( \lambda \)**: Regularization parameter ensuring invertibility of \( H + \lambda I \).
- **\( I \)**: Identity matrix.

### Cluster Score (CS)

The Cluster Score, using an Upper Confidence Bound (UCB) method, is given by:

\[
CS_i = \bar{I}_i + \alpha \sqrt{\frac{2 \ln \left(\sum_{j} T(C_j)\right)}{T(C_i)}}
\]

**Variables Breakdown:**

- **\( CS_i \)**: Score for cluster \( C_i \).
- **\( \bar{I}_i \)**: Average influence score within cluster \( C_i \).
- **\( T(C_i) \)**: Number of times data is sampled from cluster \( C_i \).
- **\( \sum_{j} T(C_j) \)**: Total number of samples across all clusters.
- **\( \alpha \)**: Hyperparameter balancing exploration and exploitation.

### Kronecker Product for Attention Layers

The Hessian matrix for the query, key, and value (QKV) layers is approximated using:

\[
H_{qkv} = E\left(\delta_{qkv}\delta_{qkv}^T\right) \otimes E\left(x_{qkv}x_{qkv}^T\right)
\]

**Variables Breakdown:**

- **\( H_{qkv} \)**: Approximate Hessian for the QKV layers.
- **\( \delta_{qkv} \)**, **\( x_{qkv} \)**: Gradient and input vectors for QKV layers respectively.
- **\( E(\cdot) \)**: Expectation of the outer product.
- **\( \otimes \)**: Kronecker product combining matrices efficiently.

---

These key formulas illustrate the balance of quality through influence functions and diversity through data exploration in the Quad method. This approach ensures that selected data for pretraining large language models is both influential and diverse, thereby enhancing model performance.

## Training Flow

1. **Problem Definition**: Identify a subset \( D_b \) from a large candidate pool \( D_c \) with reference to dataset \( D_r \).
2. **Quad Method Application**:
   - **Clustering**: Group candidate data \( D_c \) to maximize within-group similarity and cross-group diversity.
   - **Score Calculation**: Estimate influence scores within each cluster.
3. **Multi-Armed Bandit Framework**:
   - Treat clusters as "arms."
   - Select clusters with high Upper Confidence Bound (UCB).
   - Calculate and update influence scores within selected clusters.
4. **Iterative Refinement**: Update cluster scores and enrich the training dataset with influential samples.

**High-level Pseudocode**:

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

1. **Data Clustering**: Organize \( D_c \) into clusters.
2. **Multi-Armed Bandit**: Use clusters as arms to compute Cluster Scores.
3. **Influence Calculation**:
   - Compute influence scores for chosen clusters.
   - Utilize advanced methods like Kronecker product for efficiency.
4. **Sample and Select**:
   - Refine selection based on influence.
   - Maintain diversity by sampling less frequently chosen clusters.
5. **Training**: Use influential data to enhance the model.

**Inference Pseudocode**:

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

## Experiments Overview

- **Annotation Budget Ablations** (Table 5)
- **Time Cost Comparison** for Sampling Thresholds (Table 11)
- **Influence Calculation Accuracy** on MLP vs Attention Layers (Figure 4b)
- **Candidate Score vs Performance** (Figure 4c)
- **Influence Threshold Ablation Study** (Figure 4d)
- **MAB vs Top-K Clusters Exploration** (Figure 4a)
- **Openwebmath Continuous Pretraining** (Figure 7d)
- **Clustering Algorithm Impact** (Figure 5c)
- **Cluster Number Analysis** (Figure 5d)
- **Efficiency of Quad vs Direct Training** (Figure 7c)
- **Min-Max-Mean Results** Across Seeds (Tables 6, 7)

## Proofs

- **Submodularity of Influence Function**: Demonstrates efficient selection of diverse influencing data points.
- **Lower Bound Termination Guarantee**: Ensures the algorithm terminates with sufficient performance improvement.
- **Time Complexity Analysis**: Highlights the computational efficiency of the influence score calculation method.

These sections collectively describe the Quad method's approach to selecting primarily impactful and diverse datasets for pretraining large language models.