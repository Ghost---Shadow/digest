# ADAPT-∞: Scalable Continual Multimodal Instruction Tuning via Dynamic Data Selection

## Meta

* **Name**: ADAPT-∞: Scalable Continual Multimodal Instruction Tuning via Dynamic Data Selection
* **Journal**: Published as a conference paper at ICLR
* **Year**: 2025
* **Author**: Department of Computer Science, UNC Chapel Hill
* **Code**: [GitHub Repository](https://github.com/adymaharana/adapt-inf)
* **One-liner**: The paper introduces Adapt-∞, a multi-way adaptive data selection method to efficiently train large multimodal language models by dynamically balancing data selection.
* **Model**: LLaVA 1.5, Lesser models like TinyLLaVA are used for efficiency analysis
* **Datasets**: LLaVA-1.5, M3IT, MiniGPT4, MANTIS, LAMM, VisionFLAN for the main experiments, and MMLU for evaluation
* **Baselines**: Multi-task training, Sequential training, Random Experience Replay, Score-based methods (e.g., Perplexity, EL2N), SemDeDup, Density-based Pruning, COINCIDE

## Formulas

Below is a detailed breakdown of each variable and component in the formulas using MathJax-style LaTeX formatting.

---

### 1. Perplexity of a Multimodal Data Instance

The formula given is:

$$
\text{PPL}(z_i) = \exp\left(\frac{1}{|z_i|} \sum_{e_j \in z_i} \text{NLL}(e_j)\right),
$$

with:

$$
\text{NLL}(e_j) = -\log\left( P(e_j|e_{<j}, I;\theta) \right).
$$

**Breakdown**:

- $z_i$: Denotes a multimodal data instance, which is usually a sequence of tokens that may include both image-related tokens and text tokens.
- $|z_i|$: Represents the length (i.e., the total number of tokens or elements) in instance $z_i$. It acts as a normalization factor for the sum of negative log-likelihoods.
- $e_j \in z_i$: Here, $e_j$ is the $j$th element (or token) in the data instance $z_i$. The notation $e_j \in z_i$ indicates consideration of every token in the sequence.
- $\text{NLL}(e_j)$: Stands for the negative log-likelihood of token $e_j$. It quantifies the “cost” or “surprise” of seeing the token $e_j$ given the context.
- $P(e_j|e_{<j}, I; \theta)$: The probability of generating token $e_j$ conditioned on previous tokens, image tokens, and model parameters.
- $\exp(\cdot)$: The exponential function converts the average negative log-likelihood back to a perplexity score, expressing model confidence.

---

### 2. Image Grounding Score

The image grounding score is defined as:

$$
\text{IG}(z_i) = \frac{\text{PPL}(e)}{\text{PPL}(e, I)}.
$$

**Breakdown**:

- $\text{IG}(z_i)$: Image grounding score for multimodal data instance $z_i$. Measures the influence of image $I$ for explaining the text.
- $\text{PPL}(e)$: Perplexity of generating text tokens $e$ without conditioning on the image.
- $\text{PPL}(e, I)$: Perplexity of generating text tokens when conditioned on image $I$ as well.

The ratio compares the two perplexities to assess the importance of visual context.

---

### 3. Lifelong Multimodal Instruction Tuning Objective

The optimization problem is formulated as:

$$
\arg\min_\theta \frac{1}{T + 1} \sum_{t=0}^T \sum_{i=0}^{\hat{N}_t - 1} L\!\Big(f\big(\hat{x}_i^t, \hat{p}_i^t; \theta\big),\, \hat{y}_i^t\Big),
$$

subject to:

$$
T \cdot (\hat{N}_t - 1) \leq \tau.
$$

**Breakdown**:

- $\theta$: Model parameters being optimized.
- $T$: Total number of observed tasks or datasets over time.
- $\hat{N}_t$: Number of data instances in the $t$th task.
- $\hat{x}_i^t$ and $\hat{p}_i^t$: Input data components for task $t$.
- $f(\hat{x}_i^t, \hat{p}_i^t; \theta)$: Model’s output for given input and parameters.
- $\hat{y}_i^t$: Target output for input at index $i$ for task $t$.
- $L(\cdot, \cdot)$: Loss function measuring the distance between the model’s output and the true label.
- $\tau$: Predefined computational budget constraint.

---

### 4. Entropy-based Multi-Way Data Selection Within Clusters

The formula is:

$$
\hat{s}^{(k)} = \arg\max_{s^n} H\Big(\hat{P}_n^{\theta}\Big), \quad \text{where} \quad \hat{P}_n^{\theta} = \{\hat{p}_b(s^n)\}, \ \forall b \in B^{(k)}.
$$

**Breakdown**:

- $\hat{s}^{(k)}$: Selected scoring function for the $k$th cluster, maximizing entropy.
- $s^n$: Candidate scoring function for data samples.
- $\arg\max_{s^n}$: Notation for finding the $s^n$ that maximizes the function.
- $H(\hat{P}_n^{\theta})$: Entropy of distribution $\hat{P}_n^{\theta}$, measuring uncertainty.
- $\hat{P}_n^{\theta}$: Set of scores as computed under parameters $\theta$.
- $B^{(k)}$: Collection of branches or sub-clusters within the $k$th main cluster.

---

## Training Flow

### Training Flow

1. Begin with a pre-trained Multimodal Large Language Model (MLLM).
2. Integrate new datasets into the training pool.
3. Extract gradient vectors for each sample.
4. Form pseudo-task clusters using k-means on gradient vectors.
5. Evaluate sample importance using a pool of scoring function experts.
6. Train the model on selected samples while maintaining a computational budget.
7. Prune semantically redundant samples from larger clusters.
8. Iterate the process with new datasets.

## Inference Flow

### Inference Flow

1. **Initial Setup**: Begin with a pre-trained MLLM and initial dataset. Define computational budget.
2. **Task of Lifelong Instruction Tuning (LiIT)**: Train and prune datasets periodically.
3. **Adaptive Data Selection**: Combine new datasets with the existing pool and form clusters.
4. **Scoring Sample Importance**: Compute diverse importance scores and dynamically select scoring functions.
5. **Data Pruning and Training**: Select important samples, distribute training budget, and prune semantic duplicates.
6. **Dynamic Data Pruning (LITE-Adapt-∞)**: Implement adaptive, multi-way pruning for efficient training.

```python
import torch
from torch.utils.data import DataLoader

# Step 1: Define constants and initialize necessary variables
NUM_CLUSTERS = 10
TRAINING_BUDGET = 25000
COMPUTATIONAL_BUDGET = ...
pruned_data_pool = []

# Step 2: Update and integrate new datasets
def integrate_new_data(new_data, data_pool):
    data_pool.extend(new_data)
    return data_pool

# Step 3: Create pseudo-task clusters
def create_clusters(data_pool):
    gradients = extract_gradients(data_pool)
    clusters = kmeans_clustering(gradients, NUM_CLUSTERS)
    return clusters

# Step 4: Score samples and select data
def score_and_select(data_pool, clusters):
    selected_data = []
    for cluster in clusters:
        scores = compute_scores(cluster)
        top_samples = select_top_samples(cluster, scores, TRAINING_BUDGET/NUM_CLUSTERS)
        selected_data.extend(top_samples)
    return selected_data

# Step 5: Train the model on selected data
def train_model(model, selected_data):
    train_loader = DataLoader(selected_data, batch_size=...)
    for batch in train_loader:
        # Forward pass
        output = model(batch)
        # Compute loss and backward pass
        loss = compute_loss(output, batch.labels)
        loss.backward()
        # Optimization step
        optimizer.step()

# Step 6: Manage data pool size with pruning
def prune_data_pool(data_pool):
    # Compute semantic representations
    representations = feature_extractor(data_pool)
    # Prune redundant samples
    pruned_pool = prune_redundant_samples(representations, data_pool)
    return pruned_pool

# Example execution flow
new_data = load_new_dataset(timestep)
data_pool = integrate_new_data(new_data, data_pool)
clusters = create_clusters(data_pool)
selected_data = score_and_select(data_pool, clusters)
train_model(model, selected_data)
pruned_data_pool = prune_data_pool(data_pool)
```

This pseudocode captures the essence of dynamically selecting and pruning data to efficiently train a lifelong adaptive multimodal model.

## Experiments

### List of Experiments

* Annotation budget ablations (Table 5)
* Time cost comparison (Table 6)
* Pseudo-task clustering vs hidden state outputs analysis (Section 4.2, Figure 4)
* Ablation results for different pruning types (Table 3)
* Efficiency analysis with different configurations using LITE-Adapt-∞ (Section 6)
* Skill-wise breakdown of relative gains (Figure 5A)
* Visual chat skill retention example (Figures 6, 7, and 8)
* Sequential vs multitask training performance (Table 2)
* Multilingual skill recovery analysis (Section 6, Figure 9)
* Lifelong instruction tuning for language-only models (Table 8)

## Proofs

### List of Proofs

The paper primarily focuses on empirical methodologies, results, and analysis for the Adapt-∞ framework, without presenting formal mathematical proofs. It emphasizes methodologies such as the Adapt-∞ data selection method and experimental setups rather than traditional proofs.