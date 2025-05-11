# Speculative Coreset Selection for Task-Specific Fine-Tuning with STAFF

## Meta

* **Name:** STAFF: Speculative Coreset Selection for Task-Specific Fine-Tuning
* **Journal:** ICLR 2025 Conference
* **Year:** 2025
* **Authors:** Xi’an Jiaotong University, Nanyang Technological University, University of Massachusetts, Amherst
* **Code:** [GitHub Repository](https://github.com/shiningrain/STAFF)
* **One-liner:** Speculate scores from larger model from smaller model
* **Models Used:** Gemma-7b, Llama-2-13b, Mistral-Nemo-Instruct-2407
* **Datasets Used:** BioInstruct, DialogSum, WMT-19 (Kazakh-English subset)
* **Baselines Compared:** Random, GraNd, EL2N, CCS, D2 Pruning

## Formulas

### Coreset Selection Objective

The objective is formulated as:

$$
\min_{D' \subseteq D\,:\, \frac{|D'|}{|D|} \leq 1 - p} \; \mathbb{E}_{x,y \sim P} \Bigl[ L(x, y; \theta(D')) \Bigr]
$$

- **$D$**: Full dataset.
- **$D'$**: Subset (coreset) selected for fine-tuning.
- **$|D|, |D'|$**: Number of samples in $D$ and $D'$ respectively.
- **$p$**: Prune rate.
- **$P$**: Probability distribution of input-output pairs $(x,y)$.
- **$L(x, y; \theta(D'))$**: Loss function for sample $(x, y)$ with model $\theta$ fine-tuned on $D'$.
- **$\theta(D')$**: Model parameters after fine-tuning on subset $D'$.

### Speculative Score Calculation

For a sample $d$, the speculative score is calculated as:

$$
S_d^s = \left\| \nabla_\phi L\bigl(\theta_s(d)\bigr) \right\|_2
$$

- **$d$**: A data sample from $D$.
- **$S_d^s$**: Speculative score for sample $d$.
- **$\theta_s$**: Small model used for estimation.
- **$L(\theta_s(d))$**: Loss for sample $d$ using $\theta_s$.
- **$\nabla_\phi L(\theta_s(d))$**: Gradient of the loss w.r.t. parameters $\phi$.
- **$\|\cdot\|_2$**: Euclidean norm of the gradient vector.

### LLM Verification & Selection

#### Verification Result for Data Regions

For a data region $B_i$, define:

$$
V_i = \frac{\sum_{d \in B^*_i} S_d^t}{\sum_{d \in B^*_i} S_d^s}
$$

- **$B_i, B^*_i$**: Data region or subset of samples.
- **$S_d^t$**: Verification score using target model $\theta_t$.
- **$V_i$**: Ratio of summed verification to speculative scores in a region.

#### Selection Budget for a Data Region

Budget for a region $B_i$:

$$
m_B = \left\lfloor \frac{(m - |D'|)V_i}{|B|} \right\rfloor
$$

- **$m$**: Total selection budget.
- **$|D'|$**: Samples already selected.
- **$V_i$**: Verification result for region $B_i$.
- **$|B|$**: Number of samples in region $B$.
- **$\lfloor \cdot \rfloor$**: Floor function.

## Training Flow

1. **Utilize Target and Small Models:**
   - Fine-tune target LLM $\theta_t$ and a smaller model $\theta_s$.
2. **Calculate Speculative Scores:**
   - Fine-tune $\theta_s$ on full dataset $D$.
   - Compute scores $S^s$ for each sample in $D$.
3. **Divide into Regions:**
   - Split $D$ into $K$ regions based on speculative scores.
4. **Verify and Select:**
   - Verify each region on target LLM $\theta_t$.
   - Calculate verification score $V_i$ and allocate selection budget $m_B$.
   - Select samples and add to coreset $D'$ until desired prune rate.
5. **Finalize Coreset:**
   - Use $D'$ to fine-tune target LLM $\theta_t$.

### Training Flow Code Example

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

1. **Fine-Tune Small Model:**
   - Estimate speculative scores using effort score metric.
2. **Regional Data Division:**
   - Stratify dataset into $K$ regions based on scores.
3. **Sequential Regional Verification:**
   - Verify importance on target LLM.
4. **Dynamic Selection Budget Adjustment:**
   - Allocate budget based on verification scores.
5. **Compile Final Coreset:**
   - Sample from regions according to budget.

### Inference Flow Code Example

```python
def coreset_selection(dataset, small_model, target_model, prune_rate, K, verification_budget):
    # Stage 1: Speculative Score Calculation
    fine_tune(small_model, dataset)
    speculative_scores = {data: calculate_effort_score(small_model, data) for data in dataset}
    
    # Create regions based on speculative scores
    regions = stratified_sampling(speculative_scores, K)
    
    coreset = set()
    total_budget = len(dataset) * (1 - prune_rate)
    
    while regions:
        selected_region = min(regions, key=lambda r: len(r))
        verification_samples = random.sample(selected_region, min(verification_budget, len(selected_region)))
        
        # Stage 2: LLM Verification
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

### List of Experiments

- Comparison with state-of-the-art coreset methods on performance and time across various pruning rates and tasks: biology question-answering, dialogue summarization, minority language translation (Tables 1, 2).
- Ablation study examining speculative score calculation and LLM verification impact (Table 3).
- Effect of fine-tuning budget $T$ on selection results (Table 12).
- Various speculative scoring functions and their effects (Table 13).
- Verification budget $b_v$ impact analysis (Table 14).
- Effect of small model size on selection (Table 15).
- STAFF applied on larger models: Qwen2.5-32B (Table 16).
- STAFF implementation on additional datasets (Table 17).
- Layer-wise weight change distribution (Figure 5).
- Similarity assessment between small and target model-derived regions (Table 18).

## Proofs

### List of Proofs

- Submodularity of influence function.
- Lower bound termination guarantee.
- Time complexity analysis.

This markdown provides a concise yet comprehensive view of STAFF’s objectives, methodologies, experimental setups, and key findings, ensuring clarity and coherence throughout.