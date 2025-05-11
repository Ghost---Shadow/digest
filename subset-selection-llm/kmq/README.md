# Diversify and Conquer: Diversity-Centric Data Selection with Iterative Refinement

## Meta

- **Title**: Diversify and Conquer: Diversity-Centric Data Selection with Iterative Refinement
- **Journal**: Under review as a conference paper at ICLR 2025
- **Year**: 2025
- **Author**: Anonymous (authors' affiliations not provided due to double-blind review)
- **Code**: Code is submitted as supplementary materials (no specific GitHub link provided)
- **One-liner**: K-means it for diversity and then find ratio of perplexity of generated to gold as quality score
- **Model**: Llama-2-7B, Llama-3-8B, Mistral-7B
- **Datasets**: Alpaca, WizardLM
- **Baselines**: Random selection, Deita, QDIT, k-Center, kM-Closest, kM-Random, Iterative kMQ

## Formulas

### 1. Clustering Objectives

#### a. k-Center Objective
The objective minimizes the furthest distance from any data point to its nearest center:

$$
\min_{C\,\subseteq\, D,\,|C|\leq k} \ \max_{x_i \in D} \ d(x_i, C)
$$

- $D$: A set of data points.
- $d$: Distance metric $D \times D \rightarrow \mathbb{R}_{\geq 0}$.
- $C$: Set of centers selected from $D$, size at most $k$.
- $x_i$: Individual data points in $D$.
- $d(x_i, C)$: Distance of $x_i$ to the closest center in $C$.

#### b. k-Means Objective
Minimizes the sum of squared distances:

$$
\min_{C\,\subseteq\, D,\,|C|\leq k} \ \sum_{x_i \in D} d^2(x_i, C)
$$

- $d^2(x_i, C)$: Squared distance from $x_i$ to its nearest center in $C$.

### 2. Quality-Based Sampling Using k-Means-Quality (kMQ)

Sampling quality points from each cluster:

$$
\{x_1, x_2, \ldots, x_{b_j}\} \sim \text{Multinomial}\Bigl(D_j,\; \bigl\{ p(x \mid q) \bigr\}_{x \in D_j}\Bigr)
$$

- $b_j$: Number of data points sampled from the $j^\text{th}$ cluster.
- $p(x \mid q)$: Probability of selecting data point $x$ based on its quality $q$.

### 3. Iterative Data Selection Process

Calculating and updating cluster weights:

$$
s_j = \frac{1}{|D_j|} \sum_{i=1}^{|D_j|} S(x_i, y_{\text{gen}}, y_{\text{gold}})
$$

$$
w^{it}_j = \frac{s_j}{\sum_{c=1}^{k} s_c} \cdot w^{it-1}_j
$$

- $s_j$: Average quality score for the $j^\text{th}$ cluster.
- $w^{it}_j$: Weight of the $j^\text{th}$ cluster at iteration $it$.

### 4. Perplexity Scoring Function

Measures the quality in terms of language modeling performance:

$$
S(x_i, y_{\text{gen}}, y_{\text{gold}}) = - \log \left( \frac{PPL(x_i \oplus y_{\text{gen}})}{PPL(x_i \oplus y_{\text{gold}})} \right)
$$

- $PPL$: Perplexity score of a given sequence.

## Training Flow

### Training Flow Steps

1. **Initialization**:
   - Initialize training budget $b$, iteration $N$, and divide clusters using k-means-quality (kMQ).

2. **Initial Fine-tuning**:
   - Fine-tune the base model $F$ on $D'$ for one epoch.

3. **Difficulty Estimation**:
   - Compute quality scores by comparing generated responses with gold responses.

4. **Iterative Resampling**:
   - Recalculate cluster weights and select new samples for the next iteration.

5. **Final Output**:
   - Return the refined data subset $D'$ and the fine-tuned model $F^{n}$.

```python
def iterative_training(D, b, N, F, S):
    D_prime = set()
    w0 = {1/k, 1/k, ..., 1/k}
    
    for it in range(N):
        b_it = b // N
        new_subset = sample_from_weights(D - D_prime, w_prev, b_it)
        D_prime = D_prime.union(new_subset)
        
        F_n = finetune_model(F, D_prime)
        
        inference_results = run_inference(F_n, D_prime)
        scores = [calculate_score(res, S) for res in inference_results]
        cluster_scores = aggregate_scores(scores)
        
        w_cur = update_weights_based_on_scores(cluster_scores)
        
    return D_prime, F_n
```

## Experiments List

- Annotation budget ablations (Table 5)
- Comparison with different data selection methods on WizardLM dataset (Table 2)
- Impact of number of clusters (Figure 3 and Table 5)
- Performance across different datasets and methods (Tables 2, 3, 6)
- Transferability of results on different base models (Table 3)
- Visualization of low-quality cluster scores (Figure 4)
- Evaluation with different clustering hyperparameters (Appendix Table 5)
- HumanEval benchmark for code generation tasks
- Effectiveness on downstream tasks (Section 4.1)
- Impact of data encoding methods (Section 4.3)

## Proofs

### List of Proofs

- Submodularity of influence function
- Lower bound termination guarantee
- Time complexity analysis

This comprehensive formulation aims to balance diversity and quality in selecting representative datasets for fine-tuning large language models, ensuring optimal model performance through iterative refinement.