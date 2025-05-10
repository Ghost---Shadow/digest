# DELIFT: Data-Efficient Language Model Instruction Fine-Tuning

## Meta

* **Name**: DELIFT: Data-Efficient Language Model Instruction Fine-Tuning
* **Journal**: International Conference on Learning Representations (ICLR) 2025
* **Year**: 2025
* **Authors**: 1University of Illinois Urbana-Champaign, 2IBM Research
* **Code**: [GitHub link](https://github.com/agarwalishika/delift)
* **One-liner**: Proposes DELIFT, a data-efficient framework leveraging a novel utility metric for fine-tuning large language models by selecting informative data subsets at each tuning stage.
* **Models**: Llama-3.2-3B, Mistral-7B-v0.1, opt-125m, Qwen2-72B-Instruct, Phi-3-mini-128k-instruct
* **Datasets**: Mix-Instruct, P3, HotpotQA, MMLU, MT-Bench, GSM-8k, SQuAD, IBM/Government domain query rewriting dataset
* **Baselines**: Full Data, Random, SelectIT, LESS, DEFT-UCS, DELIFT (SE)

## Formulas

### 1. Pairwise Utility Metric

The primary formula is:

\[
UF_{ij} = d\bigl(GT_i,\, p(y_i \mid x_i)\bigr) - d\bigl(GT_i,\, p(y_i \mid x_i, x_j, y_j)\bigr).
\]

- \(\mathbf{UF_{ij}}\): Pairwise utility metric for sample \(i\) relative to \(j\).
- \(\mathbf{GT_i}\): Ground-truth data for sample \(i\).
- \(\mathbf{p(y_i \mid x_i)}\): Probability distribution over \(y_i\) for sample \(i\).
- \(\mathbf{p(y_i \mid x_i, x_j, y_j)}\): Probability distribution over \(y_i\) for sample \(i\) when conditioning on \((x_j, y_j)\).
- \(\mathbf{d(\cdot,\cdot)}\): Distance function measuring discrepancy between two distributions.

### 2. Distance Function Using Euclidean Norm

Using the Euclidean (\(L_2\)) norm:

\[
d\bigl(GT_i,\; p(y_i \mid \cdot)\bigr) = \bigl\|1 - p(y_i \mid \cdot)\bigr\|_2.
\]

### 3. Utility Formulation via Pointwise Mutual Information

Using Kullback–Leibler divergence:

\[
UF_{ij} = \log \frac{p(y_i \mid x_i, x_j, y_j)}{p(y_i \mid x_i)}
= \sum_{t=1}^{T} \log \left( \frac{p(y_{it} \mid x_i, x_j, y_j, y_{i,<t})}{p(y_{it} \mid x_i, y_{i,<t})} \right).
\]

### 4. Submodular Optimization Objectives

#### a) Facility Location (FL)

\[
f_{FL}(A) = \sum_{i \in D} \max_{j \in A} s_{ij}.
\]

#### b) Facility Location Mutual Information (FLMI)

\[
f_{FLMI}(A; D_T) = \sum_{i \in D} \max_{j \in A} s_{ij} + \eta \sum_{j \in A} \max_{i \in D_T} s_{ij}.
\]

#### c) Facility Location Conditional Gain (FLCG)

\[
f_{FLCG}(A \mid D_E) = \sum_{i \in D} \max\Bigl(\max_{j \in A} s_{ij} - \nu \max_{k \in D_E} s_{ik},\, 0\Bigr).
\]

## Training Flow

### Training Flow

1. **Utility Metric Calculation**: Compute pairwise utility metric \( U F_{ij} \) for all data pairs.
2. **Submodular Optimization**: Define a kernel matrix \( s_{ij} \) using utilities.
3. **Subset Selection**: Apply a greedy algorithm to select subset \( A \) that maximizes submodular objective.
4. **Fine-Tuning**: Use subset \( A \) for fine-tuning according to the specific stage.
5. **Performance Validation**: Evaluate using metrics like ROUGE, BGE, and accuracy.

### Training Flow Code (High-level Pseudocode)

```python
# Step 1: Utility Metric Calculation
for (x_i, y_i) in D:
    for (x_j, y_j) in D:
        UF_ij = calculate_utility(x_i, y_i, x_j, y_j)

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

1. **Compute Utility Matrix:** Calculate the pairwise utility metric \( U F_{ij} \).
2. **Set up Submodular Kernel:** Construct kernel matrix \( s_{ij} = \max(UF_{ij}, 0) \).
3. **Choose Submodular Objective:** Select submodular objectives based on fine-tuning stage.
4. **Greedy Subset Selection:** Apply greedy maximization.
5. **Train With Selected Subset:** Use selected subset \( A \) for fine-tuning.

### Inference Flow Code (High-Level Pseudocode)

```python
def compute_pairwise_utility(data):
    UF = np.zeros((len(data), len(data)))
    # Compute utilities
    return UF

def setup_kernel(UF):
    return np.maximum(UF, 0)

def greedy_subset_selection(data, kernel, obj_fn, budget):
    selected_indices = set()
    # Greedy selection logic
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

* **Effectiveness of DELIFT**: Evaluation on Qwen2 and Phi-3 models with Mix-Instruct and P3 datasets.
* **Task-Specific Fine-Tuning**: Performance on specialized domains.
* **Continual Fine-Tuning**: Evaluation in IBM to Government and SQuAD to HotpotQA settings.
* **Subset Size Ablation**: Performance with varying subset sizes from 5% to 50%.
* **Comparison of Fine-Tuning Methodologies**: QLoRA vs. full fine-tuning.
* **Submodular Objective Comparison**: Impact of objectives on data selection.
* **LLM-as-Judge Score Distributions**: Evaluation using Prometheus criterion.
* **Visualization of Results**: Prometheus scores distribution.
* **Annotation Budget Ablation**: Varying training data percentage.
* **Time Cost Comparison**: Discussions on efficiency.
* **Theoretical Foundation Exploration**: Utility metric as pointwise mutual information.

## Proofs

### List of Proofs

1. **Theorem 1**: Connects the utility function with pointwise mutual information.
2. **Practical Computation**: Use of a length-normalized Euclidean distance for stability.

The proofs illustrate the theoretical and computational underpinnings of the utility metric integration into the DELIFT framework.

This concise yet comprehensive documentation outlines the core aspects of DELIFT, emphasizing its novel approach to data-efficient language model fine-tuning through an innovative utility-driven selection mechanism.