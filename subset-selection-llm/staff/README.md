# STAFF: Speculative Coreset Selection for Task-Specific Fine-Tuning

## Meta

* **Name**: STAFF: Speculative Coreset Selection for Task-Specific Fine-Tuning
* **Journal**: ICLR 2025 Conference
* **Year**: 2025
* **Authors**: Xi’an Jiaotong University, Nanyang Technological University, University of Massachusetts, Amherst
* **Code**: [Github link](https://github.com/shiningrain/STAFF)
* **One-liner**: Introducing STAFF, a coreset selection method leveraging small models to efficiently guide task-specific fine-tuning of large language models.
* **Models**: Gemma-7b, Llama-2-13b, Mistral-Nemo-Instruct-2407
* **Datasets**: BioInstruct, DialogSum, WMT-19 (Kazakh-English subset)
* **Baselines**: Random, GraNd, EL2N, CCS, D2 Pruning

## Formulas

Below is a breakdown of each variable in the given formulas using MathJax-style LaTeX.

---

### 1. Coreset Selection Objective

The objective is posed as

$$
\min_{D' \subseteq D\,:\, \frac{|D'|}{|D|} \leq 1 - p} \; \mathbb{E}_{x,y \sim P} \Bigl[ L(x, y; \theta(D')) \Bigr]
$$

#### Definitions:

* $\boldsymbol{D}$: The full dataset from which samples are drawn.
* $\boldsymbol{D'}$: A selected subset (or coreset) of $D$ used for fine-tuning the language model (LLM).
* $\boldsymbol{|D|}$ and $\boldsymbol{|D'|}$: The total number of samples in $D$ and in the selected subset $D'$ respectively.
* $\boldsymbol{p}$: The prune rate. The constraint $\frac{|D'|}{|D|} \leq 1 - p$ ensures that at most a fraction $1-p$ of the original data is selected (i.e., a fraction $p$ is pruned).
* $\boldsymbol{P}$: The probability distribution over the input-output pairs $(x,y)$ from which test samples are drawn.
* $\boldsymbol{(x,y)}$: A sample drawn from the test distribution $P$. Here, $x$ might be an input (for example, a question) and $y$ its corresponding output or label.
* $\boldsymbol{L(x, y; \theta(D'))}$: The loss function measured on the sample $(x, y)$ when using the model $\theta$ that has been fine-tuned on the subset $D'$. This loss could be any task-specific loss (e.g., cross-entropy).
* $\boldsymbol{\theta(D')}$: The parameters (or version) of the LLM after training (or fine-tuning) on the coreset $D'$.

---

### 2. Speculative Score Calculation

The speculative score for a sample $d$ is given by

$$
S_d^s = \left\| \nabla_\phi L\bigl(\theta_s(d)\bigr) \right\|_2
$$

#### Variables:

* $\boldsymbol{d}$: A single data sample (typically from the dataset $D$).
* $\boldsymbol{S_d^s}$: The speculative score of sample $d$. It quantifies how much the small model's parameters would change if trained on $d$.
* $\boldsymbol{\theta_s}$: The small model, which is a relatively lightweight model used to estimate the importance of samples efficiently.
* $\boldsymbol{L\bigl(\theta_s(d)\bigr)}$: The loss computed on sample $d$ using the small model $\theta_s$.
* $\boldsymbol{\nabla_\phi L\bigl(\theta_s(d)\bigr)}$: The gradient of the loss with respect to the learnable parameters $\phi$ of the small model $\theta_s$.
* $\boldsymbol{\|\cdot\|_2}$: The Euclidean (or $\ell_2$) norm, which measures the magnitude of the gradient vector.

---

### 3. LLM Verification & Selection

**(a) Verification Result for Data Regions**

The verification result for a given data region $B_i$ is defined as

$$
V_i = \frac{\sum_{d \in B^*_i} S_d^t}{\sum_{d \in B^*_i} S_d^s}
$$

#### Terms:

* $\boldsymbol{B_i}$ or $\boldsymbol{B^*_i}$: A specific data region or subset of samples from the dataset, which has been further refined (e.g., after initial filtering or grouping).
* $\boldsymbol{S_d^t}$: The verification score for sample $d$ when evaluated on the target model $\theta_t$. This score is analogous to $S_d^s$ but is derived from the target (or larger) model.
* $\boldsymbol{S_d^s}$: The speculative score from the small model (as defined earlier).
* $\boldsymbol{V_i}$: The ratio of the summed verification scores to the summed speculative scores within data region $B^*_i$. This metric indicates how well the verification scores from the target model compare with those estimated by the small model over that region.

**(b) Selection Budget for a Data Region**

The selection budget $m_B$ for a data region is computed as follows:

$$
m_B = \left\lfloor \frac{(m - |D'|)V_i}{|B|} \right\rfloor
$$

#### Definitions:

* $\boldsymbol{m}$: The total selection budget, which is the number of samples we aim to select overall after pruning.
* $\boldsymbol{|D'|}$: The number of samples already selected into the coreset $D'$.
* $\boldsymbol{m - |D'|}$: The remaining budget available for further selection.
* $\boldsymbol{V_i}$: The verification result for the data region $B_i$ (as defined above), which reflects the relative importance of that region.
* $\boldsymbol{|B|}$: The total number of samples in the data region $B$ being considered for selection.
* $\boldsymbol{\lfloor \cdot \rfloor}$: The floor function, which rounds the computed fractional budget down to the nearest integer.

---

Overall, these expressions balance the goals of minimizing the test loss of the fine-tuned model (by selecting influential samples) and ensuring that the selected samples are both important and diverse.

## Training Flow

### Training Flow

1. Use a target LLM $\theta_t$ for task-specific fine-tuning and a smaller, analogous model $\theta_s$ from the same family.
2. **Speculative Score Calculation:**
   - Fine-tune the smaller model $\theta_s$ using the full dataset $D$.
   - Compute speculative scores $S^s$ for each data sample $d$ in $D$ based on the importance of data to the fine-tuned smaller model $\theta_s$.
3. **Region Division:**
   - Divide dataset $D$ into $K$ regions based on the speculative scores $S^s$.
4. **LLM Verification & Selection:**
   - For each region $B_i$ from the smallest sample count:
     - Randomly select a subset of size $\min\{b_v, |B_i|\}$ for verification on the target LLM $\theta_t$.
     - Calculate verification score $V_i$ reflecting the difference between speculative score and LLM verification.
     - Allocate selection budget $m_B$ proportionate to $V_i$ for selecting samples from the region.
     - Add selected samples to the final coreset $D'$.
   - Repeat until the coreset $D'$ achieves the desired pruning rate.
5. **Finalization:**
   - The selected coreset $D'$ is used for the fine-tuning of the target LLM $\theta_t$.

### Training Flow Code

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
        # Select region with min sample size
        current_region = select_min_region(regions)
        # Verification on target LLM
        V_i = verify_region(current_region, theta_t, S_s, bv)
        
        # Allocate selection budget
        m_B = allocate_budget(current_region, V_i, m - len(coreset))
        selected_samples = select_samples_from_region(current_region, m_B)
        
        # Update coreset and remaining regions
        coreset.update(selected_samples)
        regions.remove(current_region)
    
    return coreset

def fine_tune(model, data, epochs):
    # Fine-tuning logic
    pass

def calculate_speculative_scores(model, data):
    # Calculate importance scores on model
    pass

def divide_into_regions(data, scores, num_regions):
    # Stratify data based on scores
    pass

def verify_region(region, model, speculative_scores, budget):
    # Sub-sample from region and verify on target LLM
    pass

def allocate_budget(region, verification_score, remaining_budget):
    # Allocate selection budget based on verification results
    pass

def select_samples_from_region(region, budget):
    # Randomly sample from region according to budget
    pass
```

This pseudo code introduces a structured approach for using the STAFF methodology, highlighting the speculative score calculation, stratified region division, verification and selection processes, and final coreset population.

## Inference Flow

### Inference Flow

1. Fine-tune the small model on the full dataset to estimate speculative scores for each data sample, using the effort score metric to measure the changes in model parameters when learning each sample.
2. Divide the dataset into $K$ regions based on speculative scores, ensuring diverse representation across the dataset by stratified sampling.
3. Sequentially verify the regions' importance to the target LLM by randomly selecting a subset within each region, fine-tuning the target model, and calculating verification scores based on the effort involved.
4. Adjust the selection budget dynamically for each region based on verification scores, increasing allocation for important (higher verified scores) regions to optimize both importance and diversity in the final coreset.
5. Compile the final coreset by sampling from each region according to the adjusted budget allocation, effectively balancing the need for critical sample retention and comprehensive dataset representation.

### Inference Flow Code

```python
# Assume fine_tune and calculate_effort_score functions are defined

def coreset_selection(dataset, small_model, target_model, prune_rate, K, verification_budget):
    # Stage 1: Speculative Score Calculation
    fine_tune(small_model, dataset)
    speculative_scores = {data: calculate_effort_score(small_model, data) for data in dataset}
    
    # Create regions based on speculative scores
    regions = stratified_sampling(speculative_scores, K)
    
    # Placeholder for coreset
    coreset = set()
    total_budget = len(dataset) * (1 - prune_rate)
    
    while regions:
        # Select region with smallest number of samples
        selected_region = min(regions, key=lambda r: len(r))
        
        # Limit the verification to the budget
        verification_samples = random.sample(selected_region, min(verification_budget, len(selected_region)))
        
        # Stage 2: LLM Verification
        fine_tune(target_model, verification_samples)
        verification_score = sum(calculate_effort_score(target_model, data) for data in verification_samples) / len(verification_samples)
        
        # Calculate selection budget for the region
        normalized_verification_score = verification_score / speculative_scores[selected_region]
        region_budget = round((total_budget - len(coreset)) * normalized_verification_score / len(regions))
        
        # Select samples based on budget
        selected_samples = random.sample(selected_region, min(region_budget, len(selected_region)))
        coreset.update(selected_samples)
        
        # Remove selected region
        regions.remove(selected_region)
    
    return list(coreset)
```

## Experiments

### List of Experiments

* Comparison with state-of-the-art coreset selection methods for performance and time overhead across different pruning rates on three LLMs and three downstream tasks: biology question-answering, dialogue summarization, and translation of minority languages (Tables 1, 2).
* Ablation study to assess the impact of speculative score calculation and LLM verification & selection on coreset selection effectiveness (Table 3).
* Investigation of fine-tuning budget $T$ on coreset selection results to understand its effect on performance (Table 12).
* Evaluation of different speculative scoring functions to compare their influence on coreset selection (Table 13).
* Analysis of verification budget $b_v$ and its impact on selection results (Table 14).
* Study on the effectiveness of STAFF using smaller speculative models through ablation of small models with varying parameter sizes (Tables 15).
* Experiments with larger models, using STAFF to guide coreset selection on larger target models like Qwen2.5-32B (Table 16).
* Application of STAFF on additional task datasets to further establish its efficacy beyond the initially presented tasks (Table 17).
* Illustration and analysis of weight change distribution across different layers to verify sample impact (Figure 5).
* Evaluation of similarity between diverse data regions estimated by small and target models using Rand Index scores (Table 18).

## Proofs

### List of Proofs

* Submodularity of influence function
* Lower bound termination guarantee
* Time complexity analysis