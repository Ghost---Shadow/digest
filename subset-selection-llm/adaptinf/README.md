# ADAPT-∞: Scalable Continual Multimodal Instruction Tuning via Dynamic Data Selection

## Meta

* **Name** - ADAPT-∞: Scalable Continual Multimodal Instruction Tuning via Dynamic Data Selection
* **Journal** - Published as a conference paper at ICLR
* **Year** - 2025
* **Author** - Department of Computer Science, UNC Chapel Hill
* **Code** - [GitHub Repo](https://github.com/adymaharana/adapt-inf)
* **One-liner** - The paper introduces Adapt-∞, a multi-way adaptive data selection method to efficiently train large multimodal language models by dynamically balancing data selection.
* **Model** - LLaVA 1.5, with lesser models like TinyLLaVA used for efficiency analysis.
* **Datasets** - LLaVA-1.5, M3IT, MiniGPT4, MANTIS, LAMM, VisionFLAN for the main experiments, and MMLU for evaluation.
* **Baselines** - Multi-task training, Sequential training, Random Experience Replay, Score-based methods (e.g., Perplexity, EL2N), SemDeDup, Density-based Pruning, COINCIDE.

## Formulas

### 1. Perplexity of a Multimodal Data Instance

The formula is given by:

$$
\text{PPL}(z_i) = \exp\left(\frac{1}{|z_i|} \sum_{e_j \in z_i} \text{NLL}(e_j)\right),
$$

where:

$$
\text{NLL}(e_j) = -\log\bigl(P(e_j \mid e_{<j}, I; \theta)\bigr).
$$

**Explanation of the variables:**

- $ z_i $: A multimodal data instance.
- $ |z_i| $: Length of the data instance $z_i$.
- $ e_j $: The $j$-th element in sequence $z_i$.
- $ e_{<j} $: Sequence of elements preceding $e_j$ in $z_i$.
- $ I $: Image component associated with the multimodal instance.
- $ \theta $: Model parameters.
- $ P(e_j \mid e_{<j}, I; \theta) $: Probability assigned by the model to the element $e_j$ given previous elements and image $I$.
- $ \text{NLL}(e_j) $: Negative Log-Likelihood for element $e_j$.

### 2. Image Grounding Score

The image grounding score is defined as:

$$
\text{IG}(z_i) = \frac{\text{PPL}(e)}{\text{PPL}(e, I)}.
$$

**Explanation of the variables:**

- $\text{PPL}(e)$: Perplexity of the token sequence $e$ without image context.
- $\text{PPL}(e, I)$: Perplexity of sequence $e$ with image $I$ context.
- $ \text{IG}(z_i) $: Image grounding score for instance $z_i$.

### 3. Training Objective at Time Step $T$

The training objective is formulated as:

$$
\arg \min_{\theta} \frac{1}{T + 1} \sum_{t=0}^{T} \sum_{i=0}^{\hat{N}_t - 1} L\Big( f (\hat{x}_i^t, \hat{p}_i^t; \theta), \hat{y}_i^t \Big),
$$

subject to the constraint:

$$
T \cdot (\hat{N}_t - 1) \leq \tau.
$$

**Explanation of the variables:**

- $ T $: Current time step.
- $ \theta $: Model parameters.
- $ \hat{N}_t $: Selected samples at time step $ t $.
- $ N_t $: Total samples at time step $ t $.
- $ L(\cdot,\cdot) $: Loss function.
- $ f(\hat{x}_i^t, \hat{p}_i^t; \theta) $: Model prediction function.
- $ \hat{y}_i^t $: Ground truth label.
- $ \tau $: Computational budget constraint.

### 4. Selection of $\hat{s}$

The selection mechanism is given by:

$$
\hat{s}^{(k)} = \arg \max_{s_n} H \Big( \hat{P}_n^{\theta} \Big),
$$

where:

$$
\hat{P}_n^{\theta} = \big\{ \hat{p}_b^{(n)}(x) \big\}, \quad \forall\, b \in B^{(k)} \text{ and } x \in C_k.
$$

**Explanation of the variables:**

- $\hat{s}^{(k)}$: Selected sample/score for group $k$.
- $ H(\hat{P}_n^{\theta}) $: Entropy measure over probabilities $\hat{P}_n^{\theta}$.
- $\hat{P}_n^{\theta}$: Set of probabilities for sample index $n$.
- $\hat{p}_b^{(n)}(x)$: Probability for class/label $b$.
- $ B^{(k)} $: Set representing a specific block/subset.
- $ C_k $: Set of inputs/contexts for group $k$.

## Training Flow

1. Begin with a pre-trained Multimodal Large Language Model (MLLM).
2. Integrate new datasets into the training pool with previous datasets.
3. Extract gradient vectors for each dataset pool sample.
4. Perform pseudo-task clustering using k-means on gradient vectors.
5. For each pseudo-task cluster:
   - Evaluate sample importance with scoring experts.
   - Select important samples using the chosen scoring function.
6. Train the model on selected samples, respecting computational budget.
7. Measure semantic similarities and prune redundant samples.

## Inference Flow

1. **Initial Setup**: Start with a pre-trained MLLM and initial dataset pool.
2. **Task of Lifelong Instruction Tuning (LiIT)**: Include new datasets periodically.
3. **Adaptive Data Selection**: Combine new datasets with existing pool.
4. **Scoring Sample Importance**: Compute importance scores for data samples.
5. **Data Pruning and Training**: Select important samples and train.
6. **Reduce Data Pool Size**: Prune semantically redundant samples.
7. **Dynamic Data Pruning (LITE-Adapt-∞)**: Use adaptive pruning to balance efficiency and training requirements.

### Inference Flow Code

```python
import torch
from torch.utils.data import DataLoader

# Assume model, datasets, and feature_extractor are predefined

NUM_CLUSTERS = 10
TRAINING_BUDGET = 25000
COMPUTATIONAL_BUDGET = ...
pruned_data_pool = []

def integrate_new_data(new_data, data_pool):
    data_pool.extend(new_data)
    return data_pool

def create_clusters(data_pool):
    gradients = extract_gradients(data_pool)
    clusters = kmeans_clustering(gradients, NUM_CLUSTERS)
    return clusters

def score_and_select(data_pool, clusters):
    selected_data = []
    for cluster in clusters:
        scores = compute_scores(cluster)
        top_samples = select_top_samples(cluster, scores, TRAINING_BUDGET/NUM_CLUSTERS)
        selected_data.extend(top_samples)
    return selected_data

def train_model(model, selected_data):
    train_loader = DataLoader(selected_data, batch_size=...)
    for batch in train_loader:
        output = model(batch)
        loss = compute_loss(output, batch.labels)
        loss.backward()
        optimizer.step()

def prune_data_pool(data_pool):
    representations = feature_extractor(data_pool)
    pruned_pool = prune_redundant_samples(representations, data_pool)
    return pruned_pool

new_data = load_new_dataset(timestep)
data_pool = integrate_new_data(new_data, data_pool)
clusters = create_clusters(data_pool)
selected_data = score_and_select(data_pool, clusters)
train_model(model, selected_data)
pruned_data_pool = prune_data_pool(data_pool)
```

## Experiments

### List of Experiments

* Annotation budget ablations (Table 5)
* Time cost comparison (Table 6)
* Pseudo-task clustering vs hidden state outputs analysis (Section 4.2, Figure 4)
* Ablation results (Table 3)
* Efficiency analysis (Section 6)
* Skill-wise breakdown (Figure 5A)
* Visual chat skill retention (Figures 6, 7, 8)
* Sequential vs multitask training performance (Table 2)
* Recovery analysis for multilingual skill (Section 6, Figure 9)
* Experiments on lifelong instruction tuning for language-only models (Table 8)

## Proofs

The paper primarily details empirical and experimental methodologies, results, and analysis for the Adapt-∞ framework. It does not provide formal mathematical proofs but focuses on the methodologies and results of implementing Adapt-∞ in lifelong multimodal instruction tuning.