# Selective Annotation Makes Language Models Better Few-Shot Learners

## Meta

* **Name**: Selective Annotation Makes Language Models Better Few-Shot Learners
* **Journal**: International Conference on Learning Representations (ICLR)
* **Year**: 2023
* **Authors**: The University of Hong Kong, Carnegie Mellon University, Penn State University, University of Washington, Allen Institute for AI, Meta AI
* **Code**: [GitHub link](https://github.com/HKUNLP/icl-selective-annotation)
* **One Liner**: The paper presents a two-step selective annotation framework to improve few-shot learning in language models by choosing and annotating diverse and representative examples.
* **Model**: GPT-J, Codex-davinci-002, GPT-3, GPT-Neo, OPT-175B
* **Datasets**: MRPC, SST-5, DBpedia, MNLI, RTE, HellaSwag, MWoZ 2.4, GeoQuery, Natural Questions (NQ), XSum
* **Baselines**: Random selective annotation, supervised finetuning methods like RoBERTa and DS2-T5, Maximizing facility location (MFL), K-means, Diversity, Least-confidence, Conf-only, Fast vote-k.

## Formulas

Below is a detailed breakdown of each variable and symbol used in the formulas, presented using MathJax-style LaTeX.

### 1. Vote-k Selective Annotation Method

The aim is to select diverse and representative samples from a pool of unlabeled examples for annotation. This is done by forming a directed graph \( G = (V,E) \) from the document (or sentence) embeddings where:

- **\( G = (V, E) \)**:
  - **\( V \)**: Set of vertices, where each vertex is a sample represented by its Sentence-BERT embedding.
  - **\( E \)**: Set of directed edges indicating relationships or similarities between samples.

- **\( U \)**: 
  - \( U \subset V \) represents the set of candidate (unlabeled) samples chosen for annotation.

- **\( L \)**:
  - \( L \subset V \) is the set of already annotated examples.

- **\( s(v) \)**:
  - Weight assigned to a sample \( v \) based on its representation by labeled samples, defined as:
    \[
    s(v) = \rho^{-|\{\ell \in L \mid (v, \ell) \in E\}|}
    \]
  - \( |\{\ell \in L \mid (v, \ell) \in E\}| \) counts the labeled samples connected to \( v \).
  - \( \rho \) is a constant (\( \rho > 1 \)) influencing weight \( s(v) \).

- **\( \text{score}(u) \)**:
  - Computes the score of a candidate sample \( u \) as:
    \[
    \text{score}(u) = \sum_{v \in \{v \mid (v, u) \in E,\, v \in U\}} s(v)
    \]
  - Aggregates weighted votes from all vertices \( v \) that direct an edge toward \( u \).

- **Selection Process**:
  - Iteratively selects samples maximizing \( \text{score}(u) \):
    \[
    u^* = \arg\max_{u \in U} \text{score}(u)
    \]
  - Continues until a predetermined number of samples (e.g., \( M/10 \)) is reached.

- **Confidence Stratification**:
  - Splits remaining pool \( U \) into \( M \) equal-sized buckets based on model’s confidence scores, selecting additional diverse samples.

### 2. Prompt Retrieval

For each test instance, in-context examples are retrieved using:

- **\( x \) or embeddings**:
  - Text/sample embeddings using Sentence-BERT.

- **Cosine similarity, \( \cos(a, b) \)**:
  - Measures similarity between test instance \( x_{\text{test}} \) and candidate embeddings \( x_{u} \).
  
- Annotated examples with highest cosine similarities are chosen as in-context prompts.

### 3. Greedy Algorithm for Maximizing Facility Location (MFL)

Choose samples that best cover the dataset:

\[
u^* = \arg\max_{u \in U} \sum_{i=1}^{N} \left( \max \{0, \cos(x_i, x_u) - \rho_i\} \right)
\]

- **Variables**:
  - \( U \): Unlabeled candidate samples.
  - \( N \): Total number of samples concerned with representativeness.
  - \( x_i \): Embedding for the \( i^\text{th} \) sample.
  - \( x_u \): Embedding of candidate \( u \).
  - \( \rho_i \): Threshold representing similarity coverage.

- Contribution is added only when similarity exceeds the current threshold \( \rho_i \).

### 4. Additional Comparative Experiments and Evaluations

Includes:

- Comparing few-shot performance with finetuning methods.
- Measuring diversity and representativeness.
- Evaluating different annotation budgets.
  
### Summary

- **Vote-k method**: Selects samples based on weighted votes and regulates with coverage penalties.
- **Greedy MFL**: Selects samples maximizing additional representational benefits.
- Both methods leverage Sentence-BERT for similarity measurements, ensuring a diverse and representative selection for effective in-context learning.

## Training Flow

1. Formulate an annotation-efficient two-step framework.
2. Perform selective annotation:
   - Select subset \( L \) from unlabeled examples \( X = \{x_i\}_{i=1}^N \).
3. Use vote-k for selective annotation:
   - Use Sentence-BERT for embeddings.
   - Construct and evaluate k-nearest neighbor graph \( G = (V, E) \).
   - Add examples to \( L \) based on scores until \( |L| = M/10 \).
4. Generate annotations for \( L \).
5. Implement prompt retrieval at test time.
6. Evaluate few-shot learning performance.

## Inference Flow

1. Use prompt retrieval to find similar task examples.
2. Calculate embeddings and cosine similarities.
3. Select in-context examples in ascending order of similarity.
4. Generate test responses and calculate average log probability for confidence.

### Inference Code

```python
# Setup for inference using prompt retrieval
import torch
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Load language model and embedding model
model = torch.hub.load('pytorch/fairseq', 'transformer_lm', 'GPT-J')
sent_bert = SentenceTransformer('all-mpnet-base-v2')

def get_similar_examples(test_instance, labeled_examples, k=10):
    test_embedding = sent_bert.encode(test_instance)
    explicit_scores = []
    for example in labeled_examples:
        ex_embedding = sent_bert.encode(example)
        similarity = 1 - cosine(test_embedding, ex_embedding) # Cosine similarity
        explicit_scores.append((example, similarity))
    # Sort and select top K
    sorted_examples = sorted(explicit_scores, key=lambda x: x[1], reverse=True)
    return [example for example, _ in sorted_examples[:k]]

def perform_inference(test_instance, labeled_examples):
    similar_examples = get_similar_examples(test_instance, labeled_examples, k=13)
    prompt = " ".join(similar_examples) + " " + test_instance
    output_probabilities = model.generate(prompt, max_length=512, do_sample=False)
    avg_log_prob = calculate_log_prob(output_probabilities)
    return avg_log_prob

def calculate_log_prob(output_probabilities):
    log_prob_sum = sum(output_probabilities)
    return log_prob_sum / len(output_probabilities)

# Test inference
labeled_set = ["example1", "example2", "example3", ...] # Example pool
test_input = "new test instance"
confidence_score = perform_inference(test_input, labeled_set)
```

## Experiments

- **Annotation budget ablations**: Explore variance with varying budgets (Table 2).
- **Comparison of annotation methods**: Evaluate vote-k and alternatives (Table 5).
- **Domain shift effects**: Analyze robustness (Table 3).
- **Impact of model sizes**: Correlate model size to performance (Figure 3).
- **In-context learning vs finetuning**: Performance comparison (Figure 2).
- **Prompt retrieval methods**: Compare similarity-based vs random (Table 4).
- **Stability of annotation selection**: Assess across trials (Section 3.1).
- **T-SNE visualization**: Visualize selected vs full data (Figure 5).
- **Label distribution changes**: Observe shifts in annotations (Table 11).
- **Candidate selection comparison**: Various models evaluated (Figure 4).

## Proofs

The paper does not provide explicit mathematical proofs. Instead, it presents empirical evidence through experiments and evaluations to support the claims about the effectiveness of the proposed vote-k method and framework. The thorough analysis provides a compelling argument for its efficacy.