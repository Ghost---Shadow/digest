# Selective Annotation Makes Language Models Better Few-Shot Learners

## Meta

* **Name**: Selective Annotation Makes Language Models Better Few-Shot Learners
* **Journal**: International Conference on Learning Representations (ICLR)
* **Year**: 2023
* **Author**: The University of Hong Kong, Carnegie Mellon University, Penn State University, University of Washington, Allen Institute for AI, Meta AI
* **Code**: [GitHub link](https://github.com/HKUNLP/icl-selective-annotation)
* **One-liner**: The paper presents a two-step selective annotation framework to improve few-shot learning in language models by choosing and annotating diverse and representative examples.
* **Model**: GPT-J, Codex-davinci-002, GPT-3, GPT-Neo, OPT-175B
* **Datasets**: MRPC, SST-5, DBpedia, MNLI, RTE, HellaSwag, MWoZ 2.4, GeoQuery, Natural Questions (NQ), XSum
* **Baselines**: Random selective annotation, supervised fine-tuning methods like RoBERTa and DS2-T5, Maximizing Facility Location (MFL), K-means, Diversity, Least-confidence, Conf-only, Fast vote-k.

## Formulas

The paper introduces the following formulas, explained in detail below:

### Graph-based Selective Annotation

$$
\text{score}(u) = \sum_{\substack{v \in U \\ (v,u) \in E}} s(v), \quad \text{where } s(v) = \rho^{- \left| \{ \ell \in L \mid (v,\ell) \in E \} \right|}, \quad \rho > 1
$$

**Components Explained**:

- **Graph**: $ G = (V, E) $ where $ V $ are vertices (unlabeled data points $ X $) and $ E $ are the directed edges between instances.
- **Unlabeled Instances**: $ U \subset X $.
- **Annotated Instances**: $ L \subset X $ with $|L| = M$, the annotation budget.
- **Score Calculation**: 
  - $ \text{score}(u) $ is the score for an unlabeled instance $ u $.
  - $ s(v) $ is the weight function: $ s(v) = \rho^{-|\{ \ell \in L \mid (v,\ell) \in E \}|} $.
  - The decay factor $ \rho > 1 $ reduces the weight of nodes with many annotated neighbors.
- **Selecting Instances**: Choose $ u $ with the highest score in $ U $ for annotation.

### Confidence-based Selective Annotation

$$
\text{Confidence}(u) = \frac{1}{q} \sum_{t} \log p(q_t \mid q_{<t}, z; \Theta)
$$

**Components Explained**:

- **Probability**: $ p(q_t \mid q_{<t}, z; \Theta) $ is the predicted probability for token $ q_t $.
- **Logarithm for Stability**: Used for numerical stability.
- **Normalization**: $ \frac{1}{q} $ averages the log-probability per token.
- **Confidence Score**: Indicates how likely the sequence $ u $ is according to the model's parameters.

## Training Flow

### Training flow

1. **Framework**: Establish an annotation-efficient, two-step framework with selective annotation and prompt retrieval.
2. **Selective Annotation**:
   - Choose $ L \subset X $ with $ |L| = M $.
   - Use vote-k method:
     - Utilize Sentence-BERT embeddings and construct a k-nearest neighbor graph.
     - Score examples to promote diversity.
     - Create $ |L| = M/10 $ diverse examples from $ U $.
   - Estimate confidence scores for unselected examples.
   - Stratify and select based on diverse confidence levels.
3. **Annotate $ L $** and proceed to test time, performing prompt retrieval using Sentence-BERT for similarity search.
4. **Evaluate** few-shot learning on diverse tasks.

## Inference Flow

### Inference flow

1. **Similarity Search**: Use Sentence-BERT to find most similar examples to test instances from $ L $.
2. **Example Selection**: Formulate a prompt of top $ K $ similar examples.
3. **Model Input**: Feed examples concatenated in order of similarity to the model.
4. **Response Generation**: Provide as many examples until reaching max token length.
5. **Confidence Scoring**: Calculate average log probability for the output.

### Inference code example

```python
import torch
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Load models
model = torch.hub.load('pytorch/fairseq', 'transformer_lm', 'GPT-J')
sent_bert = SentenceTransformer('all-mpnet-base-v2')

def get_similar_examples(test_instance, labeled_examples, k=10):
    test_embedding = sent_bert.encode(test_instance)
    scores = []
    for example in labeled_examples:
        ex_embedding = sent_bert.encode(example)
        similarity = 1 - cosine(test_embedding, ex_embedding)
        scores.append((example, similarity))
    # Sort by similarity and take top K
    sorted_examples = sorted(scores, key=lambda x: x[1], reverse=True)
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

# Example usage
labeled_set = ["example1", "example2", "example3", ...]
test_input = "new test instance"
confidence_score = perform_inference(test_input, labeled_set)
```

## Experiments

### Experiments conducted

* **Annotation budget ablations** (Table 2)
* **Comparison of annotation methods** (e.g., vote-k, Table 5)
* **Domain shift effects** (Table 3)
* **Model size impact** (Figure 3)
* **Learning methods performance**: In-context vs. fine-tuning (Figure 2)
* **Comparison of retrieval methods** (Table 4)
* **Stability analysis**: Different random trials (Section 3.1)
* **Data visualization**: t-SNE for selected vs. full data (Figure 5)
* **Label distribution changes** (Table 11)
* **Candidate selection comparison** (Figure 4)

## Proofs

### List of proofs

The paper does not provide formal proofs but provides empirical evidence through experiments. The effectiveness of the vote-k method and other approaches is demonstrated through comprehensive tests across diverse scenarios, supporting the framework's efficacy through evidence-based analysis rather than mathematical proof.