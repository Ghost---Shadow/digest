# DELIFT: DATA EFFICIENT LANGUAGE MODEL INSTRUCTION FINE-TUNING

## Meta

* **Title**: DELIFT: Data Efficient Language Model Instruction Fine-Tuning
* **Journal**: International Conference on Learning Representations (ICLR) 2025
* **Year**: 2025
* **Authors**: University of Illinois Urbana-Champaign, IBM Research
* **Code**: [GitHub link](https://github.com/agarwalishika/delift)
* **One liner**: Proposes DELIFT, a data-efficient framework leveraging a novel utility metric for fine-tuning large language models by selecting informative data subsets at each tuning stage.
* **Models**: Llama-3.2-3B, Mistral-7B-v0.1, opt-125m, Qwen2-72B-Instruct, Phi-3-mini-128k-instruct
* **Datasets**: Mix-Instruct, P3, HotpotQA, MMLU, MT-Bench, GSM-8k, SQuAD, IBM/Government domain query rewriting dataset
* **Baselines**: Full Data, Random, SelectIT, LESS, DEFT-UCS, DELIFT (SE)

## Formulas

Below is a detailed breakdown of each formula, explaining every variable using MathJax style LaTeX.

### Pairwise Utility Metric

The paper defines the pairwise utility metric as:

$$
U_{F_{ij}} = d(GT_i,\, p(y_i \mid x_i)) - d(GT_i,\, p(y_i \mid x_i,\, x_j,\, y_j))
$$

Variables include:

- $\mathbf{U_{F_{ij}}}$: Utility of including the in-context example $(x_j, y_j)$ when predicting the output for the $i$-th example.
- $\mathbf{d(\cdot, \cdot)}$: Distance function measuring divergence between two distributions.
- $\mathbf{GT_i}$: Ground truth distribution for the $i$-th data sample.
- $\mathbf{p(y_i \mid x_i)}$: Predicted probability distribution for $y_i$ given input $x_i$ without additional in-context example.
- $\mathbf{p(y_i \mid x_i,\, x_j,\, y_j)}$: Predicted probability distribution for $y_i$ given input $x_i$ augmented with in-context example $(x_j, y_j)$.

### Information-Theoretic Interpretation using KL-divergence

When using Kullback-Leibler (KL) divergence as the distance metric:

$$
U_{F_{ij}} = \log\frac{p(y_i \mid x_i,\, x_j,\, y_j)}{p(y_i \mid x_i)}
$$

Expanding this for an output $y_i = \{ y_{i_1}, y_{i_2}, \dots, y_{i_T} \}$:

$$
U_{F_{ij}} = \sum_{t=1}^{T} \log\left(\frac{p(y_{i_t} \mid x_i,\, x_j,\, y_j,\, y_{i,<t})}{p(y_{i_t} \mid x_i,\, y_{i,<t})}\right)
$$

### Practical Computation using Euclidean Distance

For computational efficiency, the paper approximates the distance using length-normalized Euclidean distance:

$$
d(GT_i,\, p(y_i \mid \cdot)) = \left\|1 - p(y_i \mid \cdot)\right\|_2
$$

### Submodular Objectives

The paper employs submodular functions for data selection tailored to different fine-tuning stages. Variants include:

#### Facility Location (FL)

$$
f_{FL}(A) = \sum_{i \in D} \max_{j \in A} s_{ij}
$$

#### Facility Location Mutual Information (FLMI)

$$
f_{FLMI}(A; D_T) = \sum_{i \in D} \max_{j \in A} s_{ij} + \eta \sum_{j \in A} \max_{i \in D_T} s_{ij}
$$

#### Facility Location Conditional Gain (FLCG)

$$
f_{FLCG}(A \mid D_E) = \sum_{i \in D} \max\left(\max_{j \in A} s_{ij} - \nu \max_{k \in D_E} s_{ik},\, 0\right)
$$

## Training Flow

### Training Flow

The training process involves:

1. **Utility Metric Calculation**: Calculate the utility metric $U_{F_{ij}}$ for all data pairs in the dataset $D$.
2. **Submodular Optimization**: Define a kernel matrix using the utilities and incorporate it into submodular functions.
3. **Subset Selection**: Use a greedy algorithm to select a subset $A$ maximizing the chosen submodular objective.
4. **Fine-Tuning**: Apply the subset $A$ for model fine-tuning.
5. **Performance Validation**: Evaluate model performance using various metrics.

### Training Flow Code (High-level Pseudocode)

```python
# Step 1: Utility Metric Calculation
for (x_i, y_i) in D:
    for (x_j, y_j) in D:
        UF_ij = calculate_utility(x_i, y_i, x_j, y_j)

# Create Kernel Matrix
s_ij = max(UF_ij, 0)

# Step 2: Submodular Optimization
if fine_tuning_stage == 'instruction_tuning':
    objective = FL(s_ij)
elif fine_tuning_stage == 'task_specific':
    objective = FLMI(s_ij, D_target)
else:
    objective = FLCG(s_ij, D_existing)

# Step 3: Subset Selection
A = set()
for _ in range(k):
    d_star = max(d for d in D if d not in A, key=lambda d: objective(A | {d}) - objective(A))
    A.add(d_star)

# Step 4: Fine-Tuning
model = fine_tune(model, subset=A)

# Step 5: Performance Validation
performance = evaluate_model(model, metrics=[ROUGE, BGE, LAJ, accuracy])
```

## Inference Flow

### Inference Flow

During inference, the process involves:

1. **Compute Utility Matrix**: Calculate $U_{F_{ij}}$ for all data pairs.
2. **Set up Submodular Kernel**: Construct a kernel matrix $s_{ij}$.
3. **Choose Submodular Objective**: Select objectives based on the stage.
4. **Greedy Subset Selection**: Select subset $A$ maximizing the objective.
5. **Train With Selected Subset**: Use the subset for fine-tuning.

### Inference Flow Code (High-Level Pseudocode):

```python
def compute_pairwise_utility(data):
    # Compute utility matrix based on in-context learning performance
    UF = np.zeros((len(data), len(data)))
    for i, (x_i, y_i) in enumerate(data):
        for j, (x_j, y_j) in enumerate(data):
            pred_i = model.predict(x_i)
            pred_with_j = model.predict(x_i, context=(x_j, y_j))
            UF[i, j] = distance(gt(y_i), pred_i) - distance(gt(y_i), pred_with_j)
    return UF

def setup_kernel(UF):
    # Construct kernel matrix ensuring non-negative utility
    return np.maximum(UF, 0)

def greedy_subset_selection(data, kernel, obj_fn, budget):
    selected_indices = set()
    for _ in range(budget):
        best_gain = -np.inf
        best_idx = None
        for i in range(len(data)):
            if i in selected_indices:
                continue
            gain = obj_fn(selected_indices | {i}, kernel)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx is not None:
            selected_indices.add(best_idx)
    return [data[i] for i in selected_indices]

def train_with_selected_subset(data, obj_fn, budget):
    UF = compute_pairwise_utility(data)
    kernel = setup_kernel(UF)
    selected_data = greedy_subset_selection(data, kernel, obj_fn, budget)
    model.fine_tune(selected_data)

# Example Usage
train_with_selected_subset(data, FL_objective, budget=0.3 * len(data))
```

## Experiments

### List of Experiments

- **Effectiveness of DELIFT in Instruction Tuning**: Evaluation with reduced data on Qwen2 and Phi-3 models using Mix-Instruct and P3 datasets (Tables 1, 2, and 3).
- **Task-Specific Fine-Tuning Results**: Performance on specialized domains with HotpotQA to MMLU and Mix-Instruct to MT-Bench and GSM-8k datasets, compared with full data (Tables 4, 5, and 6).
- **Continual Fine-Tuning Evaluation**: Assimilating new data without forgetting old knowledge; effectiveness in IBM to Government and SQuAD to HotpotQA settings (Tables 7 and 8).
- **Subset Size Ablation**: Investigating performance with varying subset sizes from 5% to 50% of the training set (Figure 2).
- **Comparison of Fine-Tuning Methodologies**: QLoRA vs. full fine-tuning on the opt-125m model (Table 9).
- **Submodular Objective Comparison**: Analyzing the impact of facility location objectives on different data selection tasks (Discussion section).
- **LLM-as-Judge Score Distributions**: Evaluation of response quality across methods using Prometheus criterion (Tables 11 and 12).
- **Visualization of Results**: Distributions of Prometheus scores across various data selection and fine-tuning methods (Appendix D).
- **Annotation Budget Ablation**: Varying the percentage of training data used in subset selection (Appendix B).
- **Time Cost Comparison**: Inferred from discussions on computational efficiency and complexity (Potential topic in discussion or future work).
- **Theoretical Foundation Exploration**: Analysis in Appendix A on utility metric as pointwise mutual information.

## Proofs

### List of Proofs

1. **Theorem 1 (Informal Statement)**: Establishes a connection between the utility function and pointwise mutual information, showing its equivalency to conditional pointwise mutual information when using KL-divergence.

2. **Practical Computation**: A practical approach using length-normalized Euclidean distance for stability over KL-divergence is provided, maintaining the essence of PMI-based formulation. 

These proofs explore the theoretical foundation and practical implications of the utility metric used in DELIFT, and they are presented in the appendix of the paper.