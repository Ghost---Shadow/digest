# ADAPT-∞: Scalable Continual Multimodal Instruction Tuning via Dynamic Data Selection

## Meta Information

* **Name**: ADAPT-∞: Scalable Continual Multimodal Instruction Tuning via Dynamic Data Selection
* **Journal**: Published as a conference paper at ICLR
* **Year**: 2025
* **Author**: Department of Computer Science, UNC Chapel Hill
* **Code**: [GitHub Repository](https://github.com/adymaharana/adapt-inf)
* **One-liner**: The paper introduces Adapt-∞, a multi-way adaptive data selection method to efficiently train large multimodal language models by dynamically balancing data selection.
* **Model**: LLaVA 1.5; lesser models like TinyLLaVA are used for efficiency analysis.
* **Datasets**: LLaVA-1.5, M3IT, MiniGPT4, MANTIS, LAMM, VisionFLAN for the main experiments, and MMLU for evaluation.
* **Baselines**: Multi-task training, Sequential training, Random Experience Replay, Score-based methods (e.g., Perplexity, EL2N), SemDeDup, Density-based Pruning, COINCIDE.

## Formulas

Below is a breakdown of each formula along with an explanation of each variable and term using MathJax-style LaTeX.

### 1. Perplexity of a Multimodal Data Instance

The formula is given by:

$$\text{PPL}(z_i) = \exp\left(\frac{1}{|z_i|} \sum_{e_j \in z_i} \text{NLL}(e_j)\right), \quad \text{where} \quad \text{NLL}(e_j) = -\log\Big(p(e_j \mid e_{<j}, I; \theta)\Big)$$

Explanation:
- $ z_i $: A multimodal data instance, which typically consists of a sequence of tokens (words or subwords) in text and possibly associated image information.
- \( |z_i| \): The length (number of tokens) in the data instance \( z_i \).
- \( e_j \in z_i \): The \(j\)-th token (or element) in the sequence \( z_i \).
- \( \text{NLL}(e_j) \): Negative Log-Likelihood of the token \( e_j \).
- \( e_{<j} \): The sequence of tokens that come before token \( e_j \).
- \( I \): The image (or image features) associated with the instance.
- \( \theta \): The set of parameters of the model.
- \( p(e_j \mid e_{<j}, I; \theta) \): The probability assigned by the model.
- \( \exp(\cdot) \): The exponential function, used here to convert the average negative log-likelihood back into perplexity.

### 2. Image Grounding Score

The image grounding score is defined as:

\[
\text{IG}(z_i) = \frac{\text{PPL}(e)}{\text{PPL}(e, I)}
\]

Explanation:
- \( \text{IG}(z_i) \): The image grounding score for the multimodal instance \( z_i \).
- \( \text{PPL}(e) \): The perplexity when only the text is used.
- \( \text{PPL}(e, I) \): The perplexity when both the text and the image are used.

### 3. Lifelong Multimodal Instruction Tuning Objective

The training objective is formulated as:

\[
\arg\min_\theta \frac{1}{T + 1} \sum_{t=0}^T \sum_{i=0}^{\hat{N}_t - 1} L\big(f(\hat{x}_i^t, \hat{p}_i^t; \theta), \hat{y}_i^t\big)
\]

subject to:

\[
T \cdot (\hat{N}_t - 1) \leq \tau
\]

Explanation:
- \( \theta \): The parameters of the multimodal model.
- \( T \): The total number of tasks or datasets.
- \( \hat{N}_t \): The number of selected samples from task \( t \).
- \( \hat{x}_i^t \): The input component of the \( i \)-th sample.
- \( \hat{p}_i^t \): The prompt associated with the sample.
- \( f(\hat{x}_i^t, \hat{p}_i^t; \theta) \): The model's forward function.
- \( \hat{y}_i^t \): The target output for the input sample.
- \( L\big(\cdot, \cdot\big) \): The loss function.
- \( \tau \): A constraint representing the computational budget.

### 4. Entropy-Based Multi-Way Data Selection within Clusters

The selection strategy is given by:

\[
\hat{s}^{(k)} = \arg\max_{s^n} H\Bigl(\hat{P}_n^{\theta}\Bigr), \quad \text{where} \quad \hat{P}_n^{\theta} = \bigl\{\hat{p}_b(s^n)\bigr\}, \quad \forall\, b \in B^{(k)}
\]

Explanation:
- \( \hat{s}^{(k)} \): The optimal scoring function chosen for the \( k \)-th cluster.
- \( s^n \): A candidate scoring function.
- \( H\Bigl(\hat{P}_n^{\theta}\Bigr) \): The entropy of the distribution of scores.
- \( \hat{P}_n^{\theta} \): A set of score values from scoring function \( s^n \).
- \( \hat{p}_b(s^n) \): The score assigned to \( b \) using \( s^n \).
- \( B^{(k)} \): The set of data points in cluster \( k \).

Overall, these formulas define mechanisms to:
- Assess how well the model predicts tokens in a multimodal context.
- Measure the contribution of the image in grounding text predictions.
- Optimize the model across tasks with computational constraints.
- Select informative data samples via entropy-based strategies.

## Training Flow

### Training Flow

1. **Initialization**: Begin with a pre-trained Multimodal Large Language Model (MLLM).
2. **Dataset Integration**: At each timestep, integrate the new dataset into the training pool.
3. **Gradient Extraction**: Extract gradient vectors from the model's layers.
4. **Pseudo-task Clustering**: Perform k-means clustering to form pseudo-skill clusters.
5. **Sample Selection**:
   - Evaluate each cluster with scoring function experts.
   - Select important samples using CCS sampling strategy.
6. **Model Training**: Train the model on the selected subset.
7. **Pruning**: Measure cosine similarities and prune redundant samples.

The selection and pruning steps ensure balanced representation and efficient resource usage, minimizing forgetting and promoting skill transfer.

## Inference Flow

### Inference Flow

1. **Setup**: Begin with a pre-trained MLLM and an initial dataset pool.
2. **Task of Lifelong Instruction Tuning (LiIT)**:
   - Train the model on new and previous datasets.
   - Prune datasets to manage load.
3. **Adaptive Data Selection**:
   - Combine new datasets with the pool.
   - Form pseudo-task clusters.
4. **Score Samples**:
   - Compute scores for data samples.
   - Dynamically choose a scoring function.
5. **Data Pruning and Training**:
   - Select samples based on scoring.
   - Allocate the training budget among clusters.
6. **Data Pool Management**:
   - Prune redundant samples.
   - Retain unique samples.
7. **Dynamic Data Pruning (LITE-Adapt-∞)**:
   - Use adaptive pruning to balance training needs.

### Inference flow Code

```python
import torch
from torch.utils.data import DataLoader

# Assume model, datasets, and feature_extractor are predefined

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

This pseudocode illustrates the essence of dynamically selecting and pruning data to efficiently train a lifelong adaptive multimodal model.

## Experiments

### List of Experiments

* Annotation budget ablations (Table 5)
* Time cost comparison (Table 6)
* Pseudo-task clustering vs hidden state outputs analysis (Section 4.2, Figure 4)
* Ablation results for different ablation types (Table 3)
* Efficiency analysis with different configurations (Section 6)
* Skill-wise breakdown of relative gains (Figure 5A)
* Visual chat skill retention over time example (Figures 6, 7, 8)
* Sequential vs multitask training performance analysis (Table 2)
* Recovery analysis for multilingual skill (Section 6, Figure 9)
* Experiments on lifelong instruction tuning for language-only models (Table 8)

These experiments assess various aspects of the Adapt-∞ approach, including data selection strategies, clustering analyses, computational efficiency, skill retention, and task performance.

## Proofs

### List of Proofs

The paper primarily details empirical and experimental methodologies, results, and analysis for the Adapt-∞ framework within lifelong multimodal instruction tuning. It focuses on methodologies, such as Adapt-∞ data selection, empirical analysis, experimental setups, and results with various datasets and models, rather than formal mathematical proofs.