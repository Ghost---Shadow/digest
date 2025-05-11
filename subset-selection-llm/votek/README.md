# Selective Annotation Makes Language Models Better Few-Shot Learners

## Meta

* **Name** - Selective Annotation Makes Language Models Better Few-Shot Learners
* **Journal** - International Conference on Learning Representations (ICLR)
* **Year** - 2023
* **Authors** - The University of Hong Kong, Carnegie Mellon University, Penn State University, University of Washington, Allen Institute for AI, Meta AI
* **Code** - [GitHub link](https://github.com/HKUNLP/icl-selective-annotation)
* **One Liner** - The paper presents a two-step selective annotation framework to improve few-shot learning in language models by choosing and annotating diverse and representative examples.
* **Model** - GPT-J, Codex-davinci-002, GPT-3, GPT-Neo, OPT-175B
* **Datasets** - MRPC, SST-5, DBpedia, MNLI, RTE, HellaSwag, MWoZ 2.4, GeoQuery, Natural Questions (NQ), XSum
* **Baselines** - Random selective annotation, supervised finetuning methods like RoBERTa and DS2-T5, Maximizing facility location (MFL), K-means, Diversity, Least-confidence, Conf-only, Fast vote-k.

## Formulas

Below is a detailed breakdown of each variable in the provided formulas using MathJax-style LaTeX.

---

1. For the selective annotation graph-based approach, the formula is:

$$
\text{score}(u) = \sum_{\substack{v \in U \\ (v,u) \in E}} s(v), \quad \text{where } s(v) = \rho^{- \left| \{ \ell \in L \mid (v,\ell) \in E \} \right|}, \quad \rho > 1
$$

Let’s break down each component:

- $G = (V, E)$:  
  - $V$ is the set of vertices in the graph representing the instances. Vertices represent the unlabeled data points $X$.  
  - $E$ denotes the set of directed edges between the instances.

- $U$:  
  - This is the set of unlabeled instances—we denote it by $U \subset X$.  
  - Only vertices in $U$ are considered while calculating scores.

- $L \subset X$:  
  - This represents the subset of instances that have been annotated.  
  - The annotation budget is given by $|L| = M$, where $M$ is the number of samples that can be annotated.

- $\text{score}(u)$:  
  - This is the overall score for an unlabeled instance $u$.  
  - It is computed as the sum of contributions $s(v)$ from neighboring vertices $v$ that point to $u$ (i.e., there is an edge $(v,u) \in E$).

- $s(v)$:  
  - This function provides a weight for node $v$.  
  - It is defined as $s(v) = \rho^{-|\{ \ell \in L \mid (v,\ell) \in E \}|}$.  
  - The set $\{ \ell \in L \mid (v,\ell) \in E \}$ contains all annotated vertices that $v$ is connected to by a directed edge.  
  - $| \cdot |$ denotes the cardinality (i.e., the number of elements) of that set.  
  - $\rho > 1$ is a hyperparameter that controls the decay; as the number of annotated neighbors increases, $s(v)$ decreases.

- $\text{argmax}_{u \in U}\, \text{score}(u)$:  
  - In each iteration, among the set $U$ of unlabeled instances, the instance $u$ with the highest score is chosen for annotation.

---

2. For the selective annotation confidence score computed using a language model, the formula is:

$$
\text{Confidence}(u) = \frac{1}{q} \sum_{t} \log p(q_t \mid q_{<t}, z; \Theta)
$$

Each variable is described as follows:

- $p(q_t \mid q_{<t}, z; \Theta)$:  
  - This represents the probability predicted by the language model for token $q_t$ given:  
    - $q_{<t}$: The sequence of tokens before time step $t$ (i.e., the context).
    - $z$: An additional input context or auxiliary information that might be provided to the model.  
    - $\Theta$: The set of parameters for the language model.
  
- $\log p(q_t \mid q_{<t}, z; \Theta)$:  
  - The logarithm of the predicted probability for token $q_t$, which is typically used for numerical stability and ease of summation.

- $\sum_{t}$:  
  - This denotes a sum over time steps $t$. It aggregates the log probabilities across the tokens composing the query or the instance $u$.

- $\frac{1}{q}$:  
  - The factor $\frac{1}{q}$ is a normalization factor, where $q$ is the length of the query or the number of tokens.  
  - This normalization yields the average log-probability per token, providing a confidence score for the instance $u$.

- $\text{Confidence}(u)$:  
  - This is the final confidence score for an instance $u$ based on the language model’s predictions.  
  - High confidence (i.e., higher average log-probability) suggests the model finds the sequence $u$ likely according to its current parameters.

---

### Contextual Overview

- In the selective annotation approach, the goal is to efficiently choose a subset $L$ of the unlabeled set $X$ such that the selected instances are both representative and diverse.
- The first formula leverages a graph-based scoring mechanism where nodes are scored based on both their connection to other unlabeled examples and their connection to already annotated instances. The decay factor $\rho$ reduces the weight of nodes that already have many annotated neighbors.
- The second formula utilizes the confidence of a language model to further assess and partition the unlabeled set $U$ by estimating how confidently the model predicts the tokens in each instance. This confidence metric helps ensure that the selected examples cover different levels of difficulty or representativeness.

Together, these methods aim to improve annotation efficiency by strategically selecting examples that are both informative (via the graph-based similarity and diversity computation) and suitably challenging (via model confidence), ultimately enhancing the performance of in-context learning models without needing an excessive amount of annotated data.

## Training Flow

### Training Flow

1. Formulate an annotation-efficient, two-step framework involving selective annotation and prompt retrieval.
2. Selective annotation is performed before the test data is available:
   - Given a set of unlabeled examples $X = \{x_i\}_{i=1}^N$, choose a subset $L \subset X$ with $|L| = M$, which represents the annotation budget.
3. Use the vote-k method for selective annotation:
   - Represent each example using Sentence-BERT embeddings.
   - Construct a k-nearest neighbor graph $G = (V, E)$.
   - Score each example based on its connections in $G$, discounting examples already similar to chosen ones to promote diversity.
   - Add examples to $L$ based on scores until $|L| = M/10$.
   - Use a language model to estimate confidence scores for examples remaining in $U$.
   - Stratify examples in $U$ based on confidence scores and select from diverse confidence levels.
4. Generate annotations for examples in $L$.
5. At test time, perform prompt retrieval:
   - Retrieve in-context examples from $L$ using Sentence-BERT to find the most similar to each test instance.
6. Evaluate few-shot learning performance using selected annotations against a set of diverse tasks.

## Inference Flow

### Inference Flow

1. At test time, use the prompt retrieval method to find the most similar task examples to the test instance from the annotated set. Calculate embeddings for all examples using Sentence-BERT and compare each test instance to determine similitude using cosine similarity.
2. Create a list of potential in-context examples for each test instance, consisting of the top $K$ most similar examples.
3. Concatenate examples in ascending order of similarity to the test instance before feeding them to the model.
4. For test response generation, feed the language model with as many examples as possible until the maximum token length is reached.
5. Calculate the average log probability over the language model’s generation output for each instance to determine its confidence score.

### Inference Flow Code

```python
# Setup for inference using prompt retrieval
import torch
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Load language model and example embedding model
model = torch.hub.load('pytorch/fairseq', 'transformer_lm', 'GPT-J')
sent_bert = SentenceTransformer('all-mpnet-base-v2')

def get_similar_examples(test_instance, labeled_examples, k=10):
    test_embedding = sent_bert.encode(test_instance)
    explicit_scores = []
    for example in labeled_examples:
        ex_embedding = sent_bert.encode(example)
        similarity = 1 - cosine(test_embedding, ex_embedding) # Cosine similarity
        explicit_scores.append((example, similarity))
    # Sort examples by similarity score and select top K
    sorted_examples = sorted(explicit_scores, key=lambda x: x[1], reverse=True)
    return [example for example, _ in sorted_examples[:k]]

def perform_inference(test_instance, labeled_examples):
    # Retrieve top K similar examples
    similar_examples = get_similar_examples(test_instance, labeled_examples, k=13)
    
    # Concatenate examples to form the prompt
    prompt = " ".join(similar_examples) + " " + test_instance
    
    # Generate output and calculate confidence score
    output_probabilities = model.generate(prompt, max_length=512, do_sample=False)
    avg_log_prob = calculate_log_prob(output_probabilities)
    
    return avg_log_prob

def calculate_log_prob(output_probabilities):
    # Placeholder function to compute average log probability
    log_prob_sum = sum(output_probabilities)
    return log_prob_sum / len(output_probabilities)

# Test the model inference
labeled_set = ["example1", "example2", "example3", ...] # Example annotated pool
test_input = "new test instance"
confidence_score = perform_inference(test_input, labeled_set)
```

## Experiments

### List of Experiments

* Annotation budget ablations (Table 2)
* Comparison of annotation methods including vote-k (Table 5)
* Domain shift effects on selective annotation methods (Table 3)
* Impact of language model sizes on annotation methods (Figure 3)
* In-context learning versus finetuning performance across tasks (Figure 2)
* Comparison of similarity-based and random prompt retrieval (Table 4)
* Evaluation of selective annotation stability with different random trials (Section 3.1)
* T-SNE data visualization of selected annotation versus full data (Figure 5)
* Changes in label distribution in selected annotations (Table 11)
* Candidate selection comparison for language models (Figure 4)

## Proofs

### List of Proofs

The paper does not explicitly list out any formal proofs as part of its main contributions or methodologies. However, it does demonstrate the effectiveness of the proposed vote-k method through empirical evidence and analyses across various experiments. The methodologies and results could be interpreted as a form of evidence to support the claims made in the paper, but they are not formal mathematical proofs. The work includes evidence-based analysis such as the comparison of different annotation methods, an exploration of various language model sizes, and tests for domain shifts which provide a comprehensive argument for the proposed framework's efficacy.