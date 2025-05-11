# Diversify and Conquer: Diversity-Centric Data Selection with Iterative Refinement

## Meta

* **Name** - Diversify and Conquer: Diversity-Centric Data Selection with Iterative Refinement
* **Journal** - Under review as a conference paper at ICLR 2025
* **Year** - 2025
* **Author** - Anonymous (authors' affiliations not provided due to double-blind review)
* **Code** - Code is submitted as supplementary materials (no specific GitHub link provided)
* **One liner** - The paper proposes a diversity-focused data selection approach using iterative refinement to enhance fine-tuning of large language models on instruction data.
* **Model** - Llama-2-7B, Llama-3-8B, Mistral-7B
* **Datasets** - Alpaca, WizardLM
* **Baselines** - Random selection, Deita, QDIT, k-Center, kM-Closest, kM-Random, Iterative kMQ

## Formulas

Below is a detailed breakdown of each variable in the formulas along with explanations in MathJax style LaTeX.

### Clustering Objectives

#### a. k-Center Objective

The first objective is defined as:

$$
\min_{C\,\subseteq\, D,\,|C|\leq k} \ \max_{x_i \in D} \ d(x_i, C)
$$

Here, the variables are:

- $ D $: A set of data points.
- $ d: D \times D \rightarrow \mathbb{R}_{\geq 0} $: A distance (or dissimilarity) metric that computes the distance between any two points in $ D $.
- $ C = \{c_1, c_2, \ldots, c_k\} \subseteq D $: The set of centers (or representative points) selected from $ D $. The size of $ C $ is at most $ k $.
- $ x_i $: An individual data point in the set $ D $.
- $ d(x_i, C) $: Defined as the distance of $ x_i $ to its closest center in $ C $, i.e.,
  
  $$
  d(x_i, C) = \min_{c_j \in C} d(x_i, c_j).
  $$

The goal of the k-center objective is to choose centers $ C $ in such a way that the furthest distance from any data point $ x_i $ to its nearest center is minimized. This ensures that every data point in $ D $ is “close” to some center, thereby covering the dataset with good representativeness.

#### b. k-Means Objective

Instead of the maximum distance, the k-means objective sums up the squared distances:

$$
\min_{C\,\subseteq\, D,\,|C|\leq k} \ \sum_{x_i \in D} d^2(x_i, C)
$$

Variables are similar as before with:

- $ d^2(x_i, C) $: The squared distance from $ x_i $ to its nearest center in $ C $. Squaring emphasizes larger distances more strongly than smaller ones.
  
The aim here is to partition the dataset into clusters by choosing centers in a way that minimizes the overall variance (or squared error) within clusters.

#### c. Cluster Assignment

For each cluster center $ c_j $, the set of points assigned to that cluster is given by:

$$
D_j = \{ x_i \in D \mid d(x_i, c_j) \leq d(x_i, c_l) \text{ for all } l \neq j,\; l = 1, \ldots, k \}
$$

Where:

- $ D_j $: The subset of data points that are closest to the center $ c_j $ compared to the other centers.
- The condition $ d(x_i, c_j) \leq d(x_i, c_l) $ for all $ l \neq j $ ensures that each point $ x_i $ is assigned to the nearest center.

### Quality-Based Sampling Using k-Means-Quality (kMQ)

Once the clusters $ D_j $ have been obtained, the paper introduces a quality-based sampling process per cluster:

$$
\{x_1, x_2, \ldots, x_{b_j}\} \sim \text{Multinomial}\Bigl(D_j,\; \bigl\{ p(x \mid q) \bigr\}_{x \in D_j}\Bigr)
$$

Variables explained:

- $ \{x_1, x_2, \ldots, x_{b_j}\} $: The set of $ b_j $ data points sampled (with replacement) from cluster $ D_j $.
- $ b_j $: The sampling budget or the number of data points to be selected from the $ j\text{th} $ cluster.
- $ \text{Multinomial}(D_j, \{ p(x \mid q) \}) $: Indicates that the sampling over the cluster $ D_j $ is done according to a multinomial distribution using probabilities $ p(x \mid q) $.
- $ p(x \mid q) $: The probability of selecting data point $ x $ given its quality $ q $. The probability is adjusted (or weighted) by the quality measure of the instance, ensuring that higher-quality samples have a higher likelihood of being drawn.

### Iterative Data Selection Process

The paper proposes an iterative procedure to update cluster weights based on quality scores. First, a normalized score for cluster $ j $ is computed:

$$
s_j = \frac{1}{|D_j|} \sum_{i=1}^{|D_j|} S(x_i, y_{\text{gen}}, y_{\text{gold}})
$$

Variables:

- $ s_j $: The average quality score for the $ j\text{th} $ cluster.
- $ |D_j| $: The number of data points in cluster $ D_j $.
- $ S(x_i, y_{\text{gen}}, y_{\text{gold}}) $: A scoring function that measures the quality of the data point $ x_i $ given two responses:
  - $ y_{\text{gen}} $: The generated (or model produced) output.
  - $ y_{\text{gold}} $: The gold (reference or ground-truth) output.

Next, the weight update for each cluster at iteration $ it $ is given by:

$$
w^{it}_j = \frac{s_j}{\sum_{c=1}^{k} s_c} \cdot w^{it-1}_j
$$

Where:

- $ w^{it}_j $: The weight of the $ j\text{th} $ cluster at iteration $ it $.
- $ w^{it-1}_j $: The weight from the previous iteration $ it-1 $ for cluster $ j $.
- $ \sum_{c=1}^{k} s_c $: The sum of the quality scores over all clusters. This normalization ensures that the updated cluster weights sum to one (or remain in a comparable scale).

This update mechanism adjusts the influence of each cluster based on its performance as evaluated by the scoring function $ S $.

### Perplexity Scoring Function

For measuring the quality in terms of language modeling performance, a perplexity-based score is used:

$$
S(x_i, y_{\text{gen}}, y_{\text{gold}}) = - \log \left( \frac{PPL(x_i \oplus y_{\text{gen}})}{PPL(x_i \oplus y_{\text{gold}})} \right)
$$

Here:

- $ S(x_i, y_{\text{gen}}, y_{\text{gold}}) $: The quality score for sample $ x_i $ given the generated answer $ y_{\text{gen}} $ and the gold answer $ y_{\text{gold}} $.
- $ PPL(\cdot) $: The perplexity score of a given sequence as computed by the relevant language model. Perplexity is a measurement of how well the model predicts a sample. Lower perplexity indicates a better model fit.
- $ x_i \oplus y_{\text{gen}} $ and $ x_i \oplus y_{\text{gold}} $: The concatenation of the input question $ x_i $ with the generated answer $ y_{\text{gen}} $ and the gold answer $ y_{\text{gold}} $, respectively. The concatenation operator $ \oplus $ signifies that the model considers the full prompt-response context when computing perplexity.

The negative logarithm serves to convert the ratio of perplexities into a score where a lower perplexity (in the numerator or a higher quality response) increases the score.

### Summary

- The clustering objectives (k-center and k-means) are designed to ensure that the selected centers represent the dataset well by capturing coverage (minimizing the maximum distance) and low variance (minimizing the sum of squared distances).
- The k-means-quality (kMQ) sampling introduces a quality measure $ p(x \mid q) $ into the sampling process, ensuring that higher quality data points within each cluster are more likely to be selected.
- The iterative data selection process computes scores $ s_j $ for each cluster based on a quality function $ S $ (which itself can be based on a perplexity ratio) and updates cluster weights $ w_j $ accordingly.
- The perplexity-based score $ S(x_i, y_{\text{gen}}, y_{\text{gold}}) $ compares the quality of generated outputs versus the gold standard using language model perplexity as an indicator.

This comprehensive formulation aims to balance diversity (through clustering) and quality (through sampling and iterative weighting) when selecting representative instruction datasets for fine-tuning large language models.

## Training Flow

### Training Flow

1. **Initialization**:
   - Initialize a training budget $b$, iteration $N$, and divide clusters.
   - Use k-means-quality (kMQ) for initial clustering and sampling to form the initial subset $D'$.

2. **Initial Fine-tuning**:
   - Fine-tune the base model $F$ on $D'$ for one epoch.
   
3. **Difficulty Estimation**:
   - Perform inference to generate responses for each prompt.
   - Compute quality scores comparing generated responses ($y_{\text{gen}}$) with gold responses ($y_{\text{gold}}$) using a scorer $S$:
   $$
   S(x_i, y_{\text{gen}}, y_{\text{gold}}) = score(x_i \oplus y_{\text{gold}}) - score(x_i \oplus y_{\text{gen}})
   $$
   - Aggregate scores to determine each cluster's quality.

4. **Iterative Resampling**:
   - Recalculate the weights of each cluster based on aggregated scores:
   $$
   s_j = \frac{1}{|D_j|} \sum_{i=1}^{|D_j|} S(x_i, y_{\text{gen}}, y_{\text{gold}})
   $$
   $$
   w^j_{it} = \frac{s_j}{\sum_{c=1}^{k} s_c} w^{j}_{it-1}
   $$
   - Use these weights to adjust selected samples, focusing on high-impact clusters.
   - Select $ \frac{b}{N} $ new instances for $(it+1)$th iteration via updated cluster weights.

5. **Fine-tuning and Iteration**:
   - Repeat steps 2 through 4, fine-tuning the model and updating $D'$ via kMQ adjustments until budget $b$ is exhausted.

6. **Final Output**:
   - Return the optimally refined data subset $D'$ and the fine-tuned model $F^{n}$.

### Training Flow Pseudocode

```python
def iterative_training(D, b, N, F, S):
    D_prime = set()
    w0 = {1/k, 1/k, ..., 1/k}  # Equally distributed initial weights
    
    for it in range(N):
        b_it = b // N
        # Select new data subset based on current weights
        new_subset = sample_from_weights(D - D_prime, w_prev, b_it)
        D_prime = D_prime.union(new_subset)
        
        # Fine-tune current model
        F_n = finetune_model(F, D_prime)
        
        # Get inference results and calculate scores
        inference_results = run_inference(F_n, D_prime)
        scores = [calculate_score(res, S) for res in inference_results]
        cluster_scores = aggregate_scores(scores) 
        
        # Update cluster weights
        w_cur = update_weights_based_on_scores(cluster_scores)
        
    return D_prime, F_n
```

This flow leverages iterative refinement based on feedback from the training process, ensuring optimal data subset selection to maximize model fine-tuning efficiency.

## Inference Flow

### Inference Flow

1. **Initialization**: 
   - Given an instruction dataset $ D $ and a budget $ b $, initialize a subset $ D' $, and equal weights for all clusters.
   - Determine the iteration budget based on the total budget divided by the number of iterations $ N $.

2. **Clustering and Initial Sampling**:
   - Use k-means clustering to group the dataset $ D $ into $ k $ clusters.
   - Perform an initial sampling using k-means-quality (kMQ), where a portion of data is sampled from each cluster according to a quality score.
   
3. **Iterative Process**:
   - For each iteration $ it $ from 1 to $ N $:
     - **Sample Selection**: Select new samples with budget $ \frac{b}{N} $ from the dataset excluding $ D' $ using the previous iteration's cluster weights.
     - **Model Fine-tuning**: Fine-tune the base model using the accumulated dataset $ D' $ over selected samples.
     - **Inference and Scoring**:
       - Perform inference on selected samples and generate completions.
       - Evaluate the quality of generated completions against gold standards.
       - Compute normalized scores for each cluster based on completion quality.
     - **Weight Adjustment**: Adjust cluster weights for next iteration, increasing weights for clusters with high-quality data.

4. **Final Selection**:
   - The iteration continues until the full budget $ b $ is utilized, yielding the final selected dataset $ D' $ and a fine-tuned model.

### Inference Flow Code

```python
def iterative_data_selection(D, b, N, base_model, scorer):
    D_prime = set()  # Selected Data Subset
    w = {i: 1/k for i in range(k)}  # Initial equal weights for k clusters
    bit = b // N  # Iteration budget
    for it in range(N):
        # Sample Selection
        new_samples = sample_with_weight(D - D_prime, bit, w)
        D_prime.update(new_samples)
        
        # Model Fine-tuning
        model = finetune_model(base_model, D_prime)
        
        # Perform Inference and Scoring
        scores = []
        for sample in new_samples:
            y_gen = model.generate(sample)
            score = scorer(sample, y_gen)
            scores.append(score)
        
        # Weight Adjustment
        cluster_scores = compute_cluster_scores(scores, new_samples)
        w = adjust_cluster_weights(w, cluster_scores)
    
    # Return final dataset and model
    return D_prime, model

def sample_with_weight(data, budget, weights):
    # Implement cluster-based sampling based on given weights
    pass

def finetune_model(model, data):
    # Implement fine-tuning logic with given data
    pass

def compute_cluster_scores(scores, samples):
    # Compute cluster-level scores from sample scores
    pass

def adjust_cluster_weights(weights, cluster_scores):
    # Adjust weights based on cluster scores
    pass
```

## Experiments

### List of Experiments

* Annotation budget ablations (Table 5)
* Comparison with different data selection methods on WizardLM dataset (Table 2)
* Comparison of iterative selection approach using different sample-scoring methods (Figure 2)
* Impact of number of clusters (Figure 3 and Table 5)
* Performance of models on Alpaca dataset across different data selection methods (Table 6)
* Transferability of results on different base models (Table 3)
* Visualization of clusters with low quality scores (Figure 4)
* Evaluation with different clustering hyperparameters (Appendix Table 5)
* HumanEval benchmark for code generation tasks (used in Section 3.2 and Table 2)
* Effectiveness of clustering methods and selection strategies on downstream tasks (Section 4.1)
* Data encoding methods impact study (Section 4.3)

## Proofs

### List of Proofs

* Submodularity of influence function
* Lower bound termination guarantee
* Time complexity analysis