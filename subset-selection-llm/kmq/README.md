# Diversify and Conquer: Diversity-Centric Data Selection with Iterative Refinement

## Meta

* **Name**: Diversify and Conquer: Diversity-Centric Data Selection with Iterative Refinement
* **Journal**: Under review as a conference paper at ICLR 2025
* **Year**: 2025
* **Author**: Anonymous (authors' affiliations not provided due to double-blind review)
* **Code**: Code is submitted as supplementary materials (no specific GitHub link provided)
* **One-liner**: The paper proposes a diversity-focused data selection approach using iterative refinement to enhance fine-tuning of large language models on instruction data.
* **Model**: Llama-2-7B, Llama-3-8B, Mistral-7B
* **Datasets**: Alpaca, WizardLM
* **Baselines**: Random selection, Deita, QDIT, k-Center, kM-Closest, kM-Random, Iterative kMQ

## Formulas

Below is a step-by-step breakdown of each variable and formula in the methodology. The explanations use MathJax-style LaTeX formatting.

1. **Overall Data and Budget**

   - \( D = \{x_1, x_2, \ldots, x_n\} \)  
     - \( D \) represents the complete large and diverse set of instruction data.
     - \( x_i \) is an individual data sample (or instruction) in \( D \), where \( i = 1, \ldots, n \).

   - \( D' \subset D \)  
     - \( D' \) is the selected subset from \( D \) for training, chosen under constraints.

   - \( b \in \mathbb{N}^+ \) with \( b = |D'| \ll |D| \)  
     - \( b \) is the budget or the number of samples allowed, represented as a positive natural number.
     - \( |D'| \) denotes the number of elements in \( D' \), much smaller than \( |D| \).

2. **Clustering Setup**

   For clustering, data is treated as points in a metric space with a distance function \( d \).

   - \( d : D \times D \rightarrow \mathbb{R}_{\geq 0} \)  
     - \( d \) is a distance metric for two data points from \( D \), resulting in a nonnegative real number.

   - \( C = \{c_1, c_2, \ldots, c_k\} \subseteq D \)  
     - \( C \) is the set of chosen centers (or representatives) for clusters.
     - \( c_j \) is the center of the \( j \)th cluster.
     - There are at most \( k \) centers (i.e., clusters).

3. **Clustering Objectives**

   A. **k-center Objective**

   For the k-center objective, minimize the worst-case (maximum) distance from any data point in \( D \) to its closest center in \( C \).

   - Distance from a point to the set of centers:  
     \[
     d(x_i, C) = \min_{c_j \in C} d(x_i, c_j)
     \]

   - Objective function for k-center:
     \[
     \min_{C \subset D, |C| \leq k} \max_{x_i \in D} d(x_i, C)
     \]

   B. **k-means Objective**

   Instead of minimizing the maximum distance, minimize the sum of squared distances.

   - Objective function for k-means:
     \[
     \min_{C \subset D, |C| \leq k} \sum_{x_i \in D} d^2(x_i, C)
     \]

4. **Defining Clusters**

   Once centers \( C \) are chosen, each data point is assigned to the cluster of the nearest center.

   - Cluster assignment for the \( j \)th center:
     \[
     D_j = \{x_i \in D \mid d(x_i, c_j) \leq d(x_i, c_l) \text{ for all } l \neq j,\; l = 1, \ldots, k\}
     \]

5. **Sampling Within Clusters (k-means-quality, kMQ)**

   After defining clusters, sample a fixed number of data points from each, considering quality.

   - Let \( b \) be the total sampling budget distributed across clusters. For a cluster \( D_j \), designate a sub-budget \( b_j \).

   - Sampling from \( D_j \):
     \[
     \{x_1, x_2, \ldots, x_{b_j}\} \sim \text{Multinomial}\Bigl(D_j, \{p(x \mid q)\}_{x \in D_j}\Bigr)
     \]

## Training Flow

1. **Initialization**
   - Initialize training budget \( b \), iteration \( N \), and divide clusters.
   - Use kMQ for initial clustering and sampling to form subset \( D' \).

2. **Initial Fine-tuning**
   - Finetune base model on \( D' \) for one epoch.

3. **Difficulty Estimation**
   - Perform inference to generate responses.
   - Compute quality scores using a scorer.

4. **Iterative Resampling**
   - Recalculate weights of clusters based on scores.
   - Adjust selected samples, focusing on high-impact clusters.

5. **Fine-tuning and Iteration**
   - Repeat fine-tuning and updating \( D' \) via kMQ adjustments until budget is exhausted.

6. **Final Output**
   - Return refined data subset \( D' \) and finetuned model.

### Pseudocode

```python
def iterative_training(D, b, N, F, S):
    D_prime = set()
    w0 = {1/k, 1/k, ..., 1/k}  # Initial weights
    
    for it in range(N):
        b_it = b // N
        new_subset = sample_from_weights(D - D_prime, w_prev, b_it)
        D_prime.update(new_subset)
        
        F_n = finetune_model(F, D_prime)
        
        inference_results = run_inference(F_n, D_prime)
        scores = [calculate_score(res, S) for res in inference_results]
        cluster_scores = aggregate_scores(scores)
        
        w_cur = update_weights_based_on_scores(cluster_scores)
        
    return D_prime, F_n
```

## Inference Flow

1. **Initialization**
   - Given dataset \( D \) and budget \( b \), initialize subset \( D' \) and weights.

2. **Clustering and Initial Sampling**
   - Use k-means for clustering.
   - Perform initial sampling using kMQ.

3. **Iterative Process**
   - For each iteration, perform sample selection, model fine-tuning, inference, and weight adjustment.

4. **Final Selection**
   - Continue until budget is utilized, yielding final dataset \( D' \) and model.

### Code

```python
def iterative_data_selection(D, b, N, base_model, scorer):
    D_prime = set()
    w = {i: 1/k for i in range(k)}
    bit = b // N
    for it in range(N):
        new_samples = sample_with_weight(D - D_prime, bit, w)
        D_prime.update(new_samples)
        
        model = finetune_model(base_model, D_prime)
        
        scores = []
        for sample in new_samples:
            y_gen = model.generate(sample)
            score = scorer(sample, y_gen)
            scores.append(score)
        
        cluster_scores = compute_cluster_scores(scores, new_samples)
        w = adjust_cluster_weights(w, cluster_scores)
    
    return D_prime, model
```

## Experiments

- **Annotation Budget Ablations**: Table 5
- **Comparison on WizardLM**: Table 2
- **Iterative Selection Approach**: Figure 2
- **Impact of Clusters**: Figure 3; Table 5
- **Performance on Alpaca Dataset**: Table 6
- **Transferability**: Table 3
- **Low Quality Clusters Visualization**: Figure 4
- **Clustering Hyperparameters**: Appendix Table 5
- **HumanEval Benchmark**: Section 3.2; Table 2
- **Effectiveness on Downstream Tasks**: Section 4.1
- **Data Encoding Impact Study**: Section 4.3

## Proofs

- **Submodularity of Influence Function**
- **Lower Bound Termination Guarantee**
- **Time Complexity Analysis**