# Representative Demonstration Selection for In-Context Learning with Two-Stage Determinantal Point Process

## Meta

* **Name**: Representative Demonstration Selection for In-Context Learning with Two-Stage Determinantal Point Process
* **Journal**: Not mentioned
* **Year**: Not mentioned (likely 2023 based on references)
* **Author**: School of Artificial Intelligence, University of Chinese Academy of Sciences; The Laboratory of Cognition and Decision Intelligence, Institute of Automation, Chinese Academy of Sciences; Meituan, Beijing; Harbin Institute of Technology, Weihai; Shanghai Artificial Intelligence Laboratory
* **Code**: Not provided
* **One-liner**: The paper introduces a two-stage Determinantal Point Process method to select a representative subset of in-context demonstrations, optimizing for quality and diversity to improve In-Context Learning.
* **Model**: GPT2-xl (1.5B), GPT-J (6B), GPT-NeoX (20B)
* **Datasets**: SST-2, TREC, CB, AGNews, DBPedia, RTE
* **Baselines**: Random, Cluster, DPP, Least Confidence (LC), Cal

## Formulas

Below is a detailed explanation of each formula along with a breakdown of the involved variables and terms using MathJax-style LaTeX:

### Influence Score Formula

The formula is given by:
\[
\text{Inf}(e, e_i) = p_{\text{LM}}(y_i \mid e, x_i) - p_{\text{LM}}(y_i \mid x_i)
\]

Definitions:
- \( e \): A demonstration (or example) that we are considering as a candidate to provide context.
- \( e_i = (x_i, y_i) \): Another example consisting of an input \( x_i \) and its output (or label) \( y_i \) on which the influence is measured.
- \( p_{\text{LM}}(y_i \mid e, x_i) \): Probability assigned by the language model (LM) to \( y_i \) given \( e \) and \( x_i \).
- \( p_{\text{LM}}(y_i \mid x_i) \): Probability for \( y_i \) given only \( x_i \).

Interpretation:
The difference quantifies how much the inclusion of demonstration \( e \) improves (or degrades) the probability of correctly predicting \( y_i \) for example \( e_i \).

### Quality Metric

The formula for the quality metric is:
\[
Q(e) = \frac{\displaystyle\sum_{e_i \in D_{\text{score}}} \text{Inf}(e, e_i)}{T}
\]

Definitions:
- \( Q(e) \): The quality score of demonstration \( e \).
- \( D_{\text{score}} \): Score set, a collection of examples \( e_i \) over which the influence of \( e \) is evaluated.
- \( \text{Inf}(e, e_i) \): Influence score of demonstration \( e \) on an example \( e_i \).
- \( T \): Size of the score set \( D_{\text{score}} \).

Interpretation:
The metric \( Q(e) \) is an average of the influence scores of \( e \) over the set \( D_{\text{score}} \).

### Determinantal Point Process (DPP) Probability Measurement

The probability assigned to a subset \( Y \subseteq A \) under DPP is defined as:
\[
P(Y) = \frac{\det(\mathbf{L}_Y)}{\det(\mathbf{L} + \mathbf{I})}
\]

Definitions:
- \( Y \): Subset of items (e.g., demonstrations) selected from \( A \).
- \( A \): Index set denoting all available items or demonstrations.
- \( \mathbf{L} \): Positive semi-definite (PSD) kernel matrix.
- \( \mathbf{L}_Y \): Submatrix of \( \mathbf{L} \) for indices in \( Y \).
- \( \mathbf{I} \): Identity matrix matching \( \mathbf{L} \).

Interpretation:
This defines a distribution over subsets \( Y \) such that subsets with higher diversity and quality are more likely to be selected.

### Selection of the Representative Subset

The selection rule for a representative subset is given by:
\[
Y_{\text{best}} = \arg\max_{Y \subseteq A,\ |Y|=k} \det(\mathbf{L}_Y)
\]

Definitions:
- \( Y_{\text{best}} \): Chosen subset of items considered most representative.
- \( k \): Desired number of items in the selected subset.
- \( \det(\mathbf{L}_Y) \): Determinant of the submatrix \( \mathbf{L}_Y \).

Interpretation:
By maximizing \( \det(\mathbf{L}_Y) \), the method ensures that the selected subset is diverse and of high quality.

### Semantic Representation Matrix

The semantic representation matrix is defined as:
\[
\mathbf{L}_S = \mathbf{s}\mathbf{s}^T
\]

Definitions:
- \( \mathbf{L}_S \): Kernel matrix encoding semantic similarities.
- \( \mathbf{s} \): Stack of semantic vectors, each representing a demonstration.

Interpretation:
This matrix captures semantic relations among demonstrations.

### Influence Diversity Matrix

The influence diversity matrix is specified by:
\[
\mathbf{L}_I = \mathbf{Q} \cdot \mathbf{II}\mathbf{II}^T \cdot \mathbf{Q}
\]

Definitions:
- \( \mathbf{L}_I \): PSD matrix incorporating both quality and influence diversity.
- \( \mathbf{Q} \): Diagonal matrix scaling contributions of each item according to its quality.
- \( \mathbf{II} \): Matrix representing influence information.

Interpretation:
This matrix reflects a combination of both the individual quality of demonstrations and the diversity of their influences.

## Training Flow

### Training Flow

1. **Initialize Training Set and Semantic Representations:**
   - Start with a training set \( D = \{e_1, e_2, \cdots, e_N\} \).
   - Encode each instance using sentence-BERT.
   - Stack these representations to form the dataset matrix \( \mathbf{sss} \).

2. **Stage One: Semantic Diversity Selection using DPP:**
   - Compute the PSD matrix for semantic diversity: \( LLL_S = \mathbf{sss} \mathbf{sss}^T \).
   - Select a candidate subset \( D_{\text{sem}} \) using DPP selection: 
   \[
   D_{\text{sem}} = \arg\max_{Y \subseteq D, |Y| = N_{\text{sem}}} \det(LLL_{Y})
   \]

3. **Score Set Selection:**
   - Randomly sample a score set \( D_{\text{score}} \).

4. **Stage Two: High-Quality and Influence Diversity Selection:**
   - Compute influence and quality scores.
   - Form influence representation matrix \( \mathbf{III} \) and quality vector \( \mathbf{QQQ} \).
   - Compute another PSD matrix incorporating quality and influence diversity: 
   \[
   LLL_I = \mathbf{QQQ} \cdot (\mathbf{III} \cdot \mathbf{III}^T) \cdot \mathbf{QQQ}
   \]
   - Select a final demonstration subset \( D_{\text{dem}} \).

## Inference Flow

### Inference Flow

1. Extract semantic representations and construct a PSD matrix for semantic diversity.
2. Apply DPP to select a subset with high semantic diversity (\(D_{\text{sem}}\)).
3. Sample a score set from remaining training instances.
4. Compute influence and quality scores for instances in \(D_{\text{sem}}\).
5. Construct another PSD matrix incorporating both influence embeddings and quality scores.
6. Use DPP to select a demonstration subset optimizing for quality and influence diversity.

### Inference Flow Code

```python
import torch
from sentence_transformers import SentenceTransformer
from dpp import dpp_select  # Hypothetical DPP function

# Parameters
N_sem = 200  # Size of candidate subset
T = 50       # Size of score set
k = 8        # Size of demonstration set

# Pretrained models
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 1: Extract semantic representations
def encode_with_sbert(instances):
    return sbert_model.encode(instances, convert_to_tensor=True)

# Step 2: Select candidate subset with semantic diversity
def select_subset_with_diversity(instances):
    semantic_matrix = torch.matmul(instances, instances.T)
    return dpp_select(semantic_matrix, N_sem)

# Step 3: Sample score set randomly from non-semantic set
def sample_score_set(total_instances, excluded_indices):
    indices = set(range(len(total_instances))) - set(excluded_indices)
    return random.sample(indices, T)

# Step 4-5: Calculate influence and quality scores
def calculate_influence_and_quality(candidate_set, score_set):
    influence_scores = []
    for candidate in candidate_set:
        individual_scores = []
        for score_instance in score_set:
            inf_score = compute_influence(candidate, score_instance)
            individual_scores.append(inf_score)
        influence_scores.append(individual_scores)
    return influence_scores

# Step 6-7: Select demonstrations via quality and influence diversity
def select_final_demonstrations(influences, qualities):
    quality_matrix = torch.matmul(influences, influences.T) * qualities
    return dpp_select(quality_matrix, k)

# Load training data
semantic_representations = encode_with_sbert(train_data)

# Obtain semantic diverse candidate subset
candidate_indices = select_subset_with_diversity(semantic_representations)
candidate_set = [train_data[i] for i in candidate_indices]

# Random score set sampling
score_set_indices = sample_score_set(train_data, candidate_indices)
score_set = [train_data[i] for i in score_set_indices]

# Computing influence and quality
influence_scores = calculate_influence_and_quality(candidate_set, score_set)
quality_scores = compute_qualities(influence_scores)

# Get final demonstration set
demonstration_set_indices = select_final_demonstrations(influence_scores, quality_scores)
demonstration_set = [candidate_set[i] for i in demonstration_set_indices]
```

Note: The functions `dpp_select`, `compute_influence`, and `compute_qualities` are placeholders requiring concrete implementations.

## Experiments

### List of Experiments

* **Impact of Semantic Diversity, Instance Quality, and Influence Diversity** (Figure 2)
* **Main Results and Comparison with Baseline Methods** (Table 1)
* **Performance Comparison with Retrieval-Based Methods and Discussion on Resource Efficiency** (Figure 3 and Table 2)
* **Effect of Three Factors: Semantic Diversity, Instance Quality, and Influence Diversity** (Table 3)
* **Order Sensitivity of Representative In-Context Demonstrations** (Figure 5)
* **Transferability of Representative Demonstrations across Different Language Models** (Figure 6)
* **Quality Assessment of Selected In-Context Demonstrations through Combinatorial Ranking** (Figure 7)
* **Effect of the Subset Size on Selection Performance** (Figure 8)

## Proofs

### List of Proofs

* **Submodularity of Influence Function**
* **Lower Bound Termination Guarantee**
* **Time Complexity Analysis**