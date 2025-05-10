# STAFF: Speculative Coreset Selection for Task-Specific Fine-Tuning

## Meta

- **Name**: STAFF: Speculative Coreset Selection for Task-Specific Fine-Tuning
- **Journal**: ICLR 2025 Conference
- **Year**: 2025
- **Authors**: Xi’an Jiaotong University, Nanyang Technological University, University of Massachusetts, Amherst
- **Code**: [Github link](https://github.com/shiningrain/STAFF)
- **One-liner**: Introducing STAFF, a coreset selection method leveraging small models to efficiently guide task-specific fine-tuning of large language models.
- **Model**: Gemma-7b, Llama-2-13b, Mistral-Nemo-Instruct-2407
- **Datasets**: BioInstruct, DialogSum, WMT-19 (Kazakh-English subset)
- **Baselines**: Random, GraNd, EL2N, CCS, D2 Pruning

## Formulas

### 1. Coreset Selection Optimization Problem

\[
\min_{D' \subseteq D: \, \frac{|D'|}{|D|} \leq 1 - p} \, \mathbb{E}_{x,y \sim P} \left[L(x, y; \theta(D'))\right]
\]

Definitions:
- \( \mathbf{D} \): Full original dataset.
- \( \mathbf{D'} \): Coreset of the full dataset \( D \).
- \(|D|\) and \(|D'|\): Number of samples of dataset \( D \) and coreset \( D' \).
- \( \mathbf{p} \): Pruning rate specifying fraction of data to remove (ensures \( \frac{|D'|}{|D|} \leq 1 - p \)).
- \( \mathbf{\mathbb{E}_{x,y \sim P}} \): Expectation over data pairs from distribution \( P \).
- \( \mathbf{L(x, y; \theta(D'))} \): Loss using model parameters \( \theta(D') \) trained on coreset \( D' \).

### 2. Effort Score Calculation

\[
S^s_d = \left\| \nabla_\phi L(\theta^s(d)) \right\|_2
\]

Definitions:
- \( \mathbf{S^s_d} \): Effort score for sample \( d \), shows loss sensitivity to parameter updates.
- \( \mathbf{\nabla_\phi L(\theta^s(d))} \): Gradient of loss with small model parameters \( \theta^s \) for \( d \).
- \( \mathbf{\|\cdot\|_2} \): \( L_2 \) norm of gradient.
- \( \mathbf{d} \): Individual data sample.
- \( \mathbf{\theta^s} \): Parameters of the small model.

### 3. Verification Result Calculation

\[
V_i = \frac{\sum_{d \in B^*_i} S^t_d}{\sum_{d \in B^*_i} S^s_d}
\]

Definitions:
- \( \mathbf{V_i} \): Verification score for region \( i \), aligns speculative with target LLM evaluation.
- \( \mathbf{B^*_i} \): Data subset indexed by \( i \).
- \( \mathbf{S^t_d} \): Verification score of \( d \) by target LLM.
- \( \mathbf{S^s_d} \): Speculative effort score of \( d \) from small model.

### 4. Selection Budget Calculation

\[
m_B = \left\lfloor \frac{(m - |D'|) \, V_i}{|B|} \right\rfloor
\]

Definitions:
- \( \mathbf{m_B} \): Selection budget for region \( B_i \).
- \( \mathbf{m} \): Total sample budget.
- \(|D'|\): Number of samples in coreset.
- \( \mathbf{V_i} \): Verification score for region \( i \).
- \(|B|\): Number of samples in region \( B_i \).
- \( \lfloor \cdot \rfloor \): Floor function ensuring integer budget.

### Summary

- **Coreset Selection**: Minimize expected loss while pruning data.
- **Effort Score**: Compute sample importance using gradients.
- **Verification**: Compare speculative and target scores for region importance.
- **Budget Allocation**: Allocate budget adjusted by verification scores for regions.

## Training Flow

### Steps

1. **Initialize**: Use target LLM \(\theta_t\) and small model \(\theta_s\).
2. **Speculative Score Calculation**:
    - Fine-tune \(\theta_s\) on dataset \(D\).
    - Compute speculative scores \(S^s\) for each \(d\) in \(D\).
3. **Region Division**:
    - Partition \(D\) into \(K\) regions by \(S^s\).
4. **LLM Verification & Selection**:
    - For each region \( B_i \):
        - Select subset for LLM verification.
        - Calculate \(V_i\).
        - Allocate and select samples based on \(V_i\).
5. **Finalize**:
    - Use the final coreset \(D'\) for fine-tuning \(\theta_t\).

### Pseudo Code

```python
def coreset_selection(D, theta_t, theta_s, p, T, K, bv):
    # Step 1: Speculative Score Calculation
    theta_s = fine_tune(theta_s, D, T)
    S_s = calculate_speculative_scores(theta_s, D)
    
    # Step 2: Region Division
    regions = divide_into_regions(D, S_s, K)
    m = len(D) * (1 - p)
    coreset = set()
    
    # Step 3: LLM Verification & Selection
    while regions:
        current_region = select_min_region(regions)
        V_i = verify_region(current_region, theta_t, S_s, bv)
        
        m_B = allocate_budget(current_region, V_i, m - len(coreset))
        selected_samples = select_samples_from_region(current_region, m_B)
        
        coreset.update(selected_samples)
        regions.remove(current_region)
    
    return coreset
```

## Inference Flow

### Steps

1. Fine-tune small model on dataset to estimate speculative scores.
2. Divide dataset into \( K \) regions by speculative scores.
3. Verify region importance to target LLM using subsets and calculate scores.
4. Adjust selection budget per region based on scores.
5. Compile coreset ensuring balance of critical samples and diversity.

### Pseudo Code

```python
def coreset_selection(dataset, small_model, target_model, prune_rate, K, verification_budget):
    # Stage 1: Speculative Score Calculation
    fine_tune(small_model, dataset)
    speculative_scores = {data: calculate_effort_score(small_model, data) for data in dataset}
    
    regions = stratified_sampling(speculative_scores, K)
    
    coreset = set()
    total_budget = len(dataset) * (1 - prune_rate)
    
    while regions:
        selected_region = min(regions, key=lambda r: len(r))
        verification_samples = random.sample(selected_region, min(verification_budget, len(selected_region)))
        
        fine_tune(target_model, verification_samples)
        verification_score = sum(calculate_effort_score(target_model, data) for data in verification_samples) / len(verification_samples)
        
        normalized_verification_score = verification_score / speculative_scores[selected_region]
        region_budget = round((total_budget - len(coreset)) * normalized_verification_score / len(regions))
        
        selected_samples = random.sample(selected_region, min(region_budget, len(selected_region)))
        coreset.update(selected_samples)
        
        regions.remove(selected_region)
    
    return list(coreset)
```

## Experiments

1. **Comparison**: Evaluate coreset selection methods on performance and time across pruning rates, LLMs, and tasks.
2. **Ablations**: Study speculative score calculation and LLM selection impact.
3. **Budget Influence**: Analyze fine-tuning budget \( T \) on results.
4. **Scoring Functions**: Compare speculative scoring functions.
5. **Verification Budget**: Impact on selection from different verification budgets.
6. **Model Scalability**: Effectiveness at varying small model sizes.
7. **Larger Models**: TEST on larger models using STAFF.
8. **Diverse Datasets**: Extend STAFF application to new datasets.

## Proofs

1. **Submodularity**: Prove submodularity of influence function.
2. **Termination Guarantee**: Establish lower bound for infrastructure termination.
3. **Complexity**: Analyze time complexity of STAFF methodology.