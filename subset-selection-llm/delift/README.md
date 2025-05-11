# DELIFT: DATA EFFICIENT LANGUAGE MODEL INSTRUCTION FINE-TUNING

## Meta

- **Name**: DELIFT: DATA EFFICIENT LANGUAGE MODEL INSTRUCTION FINE-TUNING
- **Journal**: International Conference on Learning Representations (ICLR) 2025
- **Year**: 2025
- **Authors**: University of Illinois Urbana-Champaign, IBM Research
- **Code**: [Github link](https://github.com/agarwalishika/delift)
- **One-liner**: Proposes DELIFT, a data-efficient framework leveraging a novel utility metric for fine-tuning large language models by selecting informative data subsets at each tuning stage.
- **Models**: Llama-3.2-3B, Mistral-7B-v0.1, opt-125m, Qwen2-72B-Instruct, Phi-3-mini-128k-instruct
- **Datasets**: Mix-Instruct, P3, HotpotQA, MMLU, MT-Bench, GSM-8k, SQuAD, IBM/Government domain query rewriting dataset
- **Baselines**: Full Data, Random, SelectIT, LESS, DEFT-UCS, DELIFT (SE)

## Formulas

Below is a detailed breakdown of each formula, with an explanation of every variable in MathJax style LaTeX.

---

### Pairwise Utility Metric

The paper defines the pairwise utility as:

$$
U_{F_{ij}} = d(GT_i,\, p(y_i \mid x_i)) - d(GT_i,\, p(y_i \mid x_i,\, x_j,\, y_j))
$$

- **Variables**:
  - $\mathbf{U_{F_{ij}}}$: The utility of including the in-context example $(x_j, y_j)$ when predicting the output for the $i$-th example. 
  - $\mathbf{d(\cdot, \cdot)}$: A distance function measuring the divergence (or dissimilarity) between two distributions.
  - $\mathbf{GT_i}$: The ground truth distribution (or label) for the $i$-th data sample.
  - $\mathbf{p(y_i \mid x_i)}$: The predicted probability distribution for $y_i$ given the input $x_i$ without any additional in-context example.
  - $\mathbf{p(y_i \mid x_i,\, x_j,\, y_j)}$: The predicted probability distribution for $y_i$ given the primary input $x_i$ augmented with an extra in-context example $(x_j, y_j)$.

---

### Information-Theoretic Interpretation using KL-divergence

When the distance $d(\cdot,\cdot)$ is chosen as the Kullback-Leibler (KL) divergence, the utility becomes:

$$
U_{F_{ij}} = \log\frac{p(y_i \mid x_i,\, x_j,\, y_j)}{p(y_i \mid x_i)}
$$

- **Expanding token-by-token**:

$$
U_{F_{ij}} = \sum_{t=1}^{T} \log\left(\frac{p(y_{i_t} \mid x_i,\, x_j,\, y_j,\, y_{i,<t})}{p(y_{i_t} \mid x_i,\, y_{i,<t})}\right)
$$

  - **Variables**:
    - $\mathbf{T}$: Length of the output sequence $y_i$.
    - $\mathbf{y_{i,t}}$: The token at time-step $t$ in the output sequence.
    - $\mathbf{y_{i,<t}}$: Sequence of tokens generated prior to time-step $t$.

---

### Practical Computation using Euclidean Distance

For computational efficiency, the paper approximates the distance using a length-normalized Euclidean distance:

$$
d(GT_i,\, p(y_i \mid \cdot)) = \left\|1 - p(y_i \mid \cdot)\right\|_2
$$

- **Interpretation**:
  - The term $1 - p(y_i \mid \cdot)$ serves to indicate error from the true label, assuming the ground truth $GT_i$ is a one-hot vector.

---

### Submodular Objectives

The paper employs submodular functions for data selection, tailored to different fine-tuning stages. Below are the variations:

#### Facility Location (FL)

$$
f_{FL}(A) = \sum_{i \in D} \max_{j \in A} s_{ij}
$$

- **Variables**:
  - $\mathbf{A}$: A selected subset of the dataset.
  - $\mathbf{D}$: The complete dataset.
  - $\mathbf{s_{ij}}$: A similarity score between samples $i$ and $j$.

#### Facility Location Mutual Information (FLMI)

$$
f_{FLMI}(A; D_T) = \sum_{i \in D} \max_{j \in A} s_{ij} + \eta \sum_{j \in A} \max_{i \in D_T} s_{ij}
$$

- **Variables**:
  - $\mathbf{D_T}$: Target dataset.
  - $\mathbf{\eta}$: Trade-off hyperparameter.

#### Facility Location Conditional Gain (FLCG)

$$
f_{FLCG}(A \mid D_E) = \sum_{i \in D} \max\left(\max_{j \in A} s_{ij} - \nu \max_{k \in D_E} s_{ik},\, 0\right)
$$

- **Variables**:
  - $\mathbf{D_E}$: Set of examples already explained or selected.
  - $\mathbf{\nu}$: Hyperparameter controlling previously explained information.

---

In summary, these formulas provide a means to measure the direct impact (utility) of adding in-context examples, relate this utility to pointwise mutual information, and employ practical approximations for efficient data selection and model fine-tuning in DELIFT.

## Training Flow

### Training Flow

1. **Utility Metric Calculation**: Compute the pairwise utility metric $ U_{F_{ij} } $ for all data pairs $(x_i, y_i)$ and $(x_j, y_j)$ in the initial dataset $ D $.

2. **Submodular Optimization**: Define a kernel matrix $ s_{ij} $ using the utilities, e.g., $ s_{ij} = \max(U_{F_{ij}}, 0) $, and use it in submodular functions like FL, FLMI, and FLCG.

3. **Subset Selection**: Apply a greedy algorithm to select a subset $ A $ of size $ k $ that maximizes the chosen submodular objective.

4. **Fine-Tuning**: Use the selected subset $ A $ to fine-tune the language model according to the specific requirements of the stage.

5. **Performance Validation**: Evaluate model performance using metrics like ROUGE, BGE, LAJ, and classification accuracy.

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

1. **Compute Utility Matrix:** Calculate the pairwise utility metric $ U_{F_{ij} } $ for all data pairs $(x_i, y_i)$ and $(x_j, y_j)$.

2. **Set up Submodular Kernel:** Construct a kernel matrix $ s_{ij} = \max(UF_{ij}, 0) $.

3. **Choose Submodular Objective:** Select suitable submodular objectives based on the fine-tuning stage.

4. **Greedy Subset Selection:** Apply greedy maximization to select a subset $ A $ of data points.

5. **Train With Selected Subset:** Use the selected subset $ A $ for efficient fine-tuning.

### Inference Flow Code (High-Level Pseudocode):

```python
def compute_pairwise_utility(data):
    UF = np.zeros((len(data), len(data)))
    for i, (x_i, y_i) in enumerate(data):
        for j, (x_j, y_j) in enumerate(data):
            pred_i = model.predict(x_i)
            pred_with_j = model.predict(x_i, context=(x_j, y_j))
            UF[i, j] = distance(gt(y_i), pred_i) - distance(gt(y_i), pred_with_j)
    return UF

def setup_kernel(UF):
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
- **Task-Specific Fine-Tuning Results**: Performance on specialized domains from HotpotQA to MMLU and Mix-Instruct to MT-Bench and GSM-8k datasets (Tables 4, 5, and 6).
- **Continual Fine-Tuning Evaluation**: Assimilating new data without forgetting old knowledge; tested in IBM to Government and SQuAD to HotpotQA settings (Tables 7 and 8).
- **Subset Size Ablation**: Investigating performance with varying subset sizes from 5% to 50% of the training set (Figure 2).
- **Comparison of Fine-Tuning Methodologies**: QLoRA vs. full fine-tuning on the opt-125m model (Table 9).
- **Submodular Objective Comparison**: Analyzing the impact of facility location objectives on different data selection tasks.
- **LLM-as-Judge Score Distributions**: Evaluation of response quality across methods using Prometheus criterion (Tables 11 and 12).
- **Visualization of Results**: Distributions of Prometheus scores across various data selection and fine-tuning methods (Appendix D).
- **Annotation Budget Ablation**: Varying the percentage of training data used in subset selection (Appendix B).
- **Theoretical Foundation Exploration**: Analysis in Appendix A on utility metric as pointwise mutual information.

## Proofs

### List of Proofs

1. **Theorem 1 (Informal Statement)**: Establishes a connection between the proposed utility function and pointwise mutual information. When using the Kullback-Leibler (KL) divergence, the utility function $ U_{F_{ij} } $ is equivalent to the conditional pointwise mutual information between $ y_i $ and $ (x_j, y_j) $ given $ x_i $.

   $$
   U F_{ij} = \log \frac{p(y_i \mid x_i, x_j, y_j)}{p(y_i \mid x_i)}.
   $$

2. **Practical Computation**: Due to potential numerical instability with KL-divergence, a practical approach using a length-normalized Euclidean distance is employed:

   $$
   d(GT_i, p(y_i \mid \cdot)) = \left\|1 - p(y_i \mid \cdot)\right\|_2.
   $$

The theoretical connections and computational considerations are further detailed in the appendix of the paper.