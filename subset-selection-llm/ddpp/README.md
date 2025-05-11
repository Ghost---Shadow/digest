# Representative Demonstration Selection for In-Context Learning with Two-Stage Determinantal Point Process

## Meta Information

* **Title**: Representative Demonstration Selection for In-Context Learning with Two-Stage Determinantal Point Process
* **Journal**: Not mentioned
* **Year**: Likely 2023 (Based on references)
* **Authors**: School of Artificial Intelligence, University of Chinese Academy of Sciences; The Laboratory of Cognition and Decision Intelligence, Institute of Automation, Chinese Academy of Sciences; Meituan, Beijing; Harbin Institute of Technology, Weihai; Shanghai Artificial Intelligence Laboratory
* **Code**: Not provided
* **One-liner**: The paper introduces a two-stage Determinantal Point Process method to select a representative subset of in-context demonstrations, optimizing for quality and diversity to improve In-Context Learning.
* **Models Used**: GPT-2-xl (1.5B), GPT-J (6B), GPT-NeoX (20B)
* **Datasets**: SST-2, TREC, CB, AGNews, DBPedia, RTE
* **Baselines**: Random, Cluster, DPP, Least Confidence (LC), Cal

## Formulas and Concepts

### Influence of a Demonstration

The influence of a demonstration $e$ on an instance $e_i = (x_i, y_i)$ is given by:

$$
\text{Inf}(e, e_i) = p_{\text{LM}}(y_i \mid e, x_i) - p_{\text{LM}}(y_i \mid x_i)
$$

where:
- $e$: A demonstration to aid language model (LM) in learning.
- $e_i = (x_i, y_i)$: An instance with input $x_i$ and label $y_i$.
- $p_{\text{LM}}(y_i \mid e, x_i)$: LM probability for $y_i$ given $e$ and $x_i$.
- $p_{\text{LM}}(y_i \mid x_i)$: Baseline LM probability without $e$.

### Quality Metric for a Demonstration

To quantify the quality of a demonstration $e$, the metric $Q(e)$ is defined as:

$$
Q(e) = \frac{\sum_{e_i \in \mathcal{D}_{\text{score}}} \text{Inf}(e, e_i)}{T}
$$

where:
- $\mathcal{D}_{\text{score}}$: Score set used for measuring demonstration quality.
- $T$: Number of elements in the score set.
- $\sum_{e_i \in \mathcal{D}_{\text{score}}} \text{Inf}(e, e_i)$: Total influence of $e$ on the score set.

### Influence Embedding and Quality Representation

For an item $x_j$ in a semantic set $\mathcal{D}_{\text{sem}}$, a $T$-dimensional influence embedding $I_j$ is derived. By aggregating these embeddings, we obtain:

$$
\mathbf{I} \in \mathbb{R}^{N_{\text{sem}} \times T}
$$

and a quality representation vector:

$$
\mathbf{Q} \in \mathbb{R}^{T}
$$

### Combining Influence and Quality into a PSD Matrix

The overall influence representation is captured through constructing a matrix $\mathbf{L}_I$:

$$
\mathbf{L}_I = \mathbf{Q} \cdot \mathbf{I} \, \mathbf{I}^T \cdot \mathbf{Q}
$$

## Training Flow

1. **Initialization**:
   - Start with training set $D = \{e_1, e_2, \cdots, e_N\}$.
   - Use sentence-BERT for semantic representations.

2. **Stage One - Semantic Diversity**:
   - Compute PSD matrix for semantic diversity.
   - Select a candidate subset $D_{\text{sem}}$ using DPP.

3. **Score Set Selection**:
   - Randomly sample a score set $D_{\text{score}}$ from $D \setminus D_{\text{sem}}$.

4. **Stage Two - High-Quality and Influence Diversity**:
   - Compute influence and quality scores for ${D_{\text{sem}}}$.
   - Form influence matrix $\mathbf{III}$ and quality vector $\mathbf{QQQ}$.
   - Compute another PSD matrix incorporating quality and influence diversity.
   - Select final demonstration subset $D_{\text{dem}}$.

## Inference Flow

1. Extract semantic representations using sentence-BERT.
2. Apply DPP for a semantically diverse subset selection ($D_{\text{sem}}$).
3. Randomly sample a score set.
4. Compute influence and quality scores.
5. Construct PSD matrix with influence and quality.
6. Use DPP for final demonstration selection maximizing quality and diversity.

## Experiments

* Analyzed the impact of semantic diversity and quality (Figure 2).
* Compared main results with baselines (Table 1).
* Discussed resource efficiency and comparison with retrieval methods (Figure 3, Table 2).
* Investigated effects of semantic diversity, instance quality, and influence diversity (Table 3).
* Explored order sensitivity and transferability of demonstrations (Figures 5, 6).
* Evaluated quality of demonstrations through ranking (Figure 7).
* Assessed effect of subset size on selection performance (Figure 8).

## Proofs

* Submodularity of influence function.
* Lower bound termination guarantee.
* Time complexity analysis.

This document presents the complete details of an innovative approach to enhancing in-context learning through a two-stage determinantal point process focusing on quality and diversity.