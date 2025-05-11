# Leveraging Exemplars for Novel Support (LENS)

## Meta

* **Name**: Finding Support Examples for In-Context Learning
* **Journal**: Not specified
* **Year**: Not specified
* **Authors**: School of Computer Science, Fudan University; Shanghai Key Laboratory of Intelligent Information Processing, Fudan University
* **Code**: [GitHub link](https://github.com/LeeSureman/ICL_Support_Example)
* **One-liner**: Use perplexity advantage for quality and cosine similarity based diversity
* **Model**: GPT-2 (specifically GPT2-L)
* **Datasets**: SST-2, SST-5, Amazon, MR, Subj, TREC, AGNews, DBPedia
* **Baselines**: Zero-shot, Random, Random & Validation, Herding, K-Center Greedy, Entropy, Least Confidence, Margin, Cal, Forgetting, GraNd, CRAIG, GradMatch, Facility Location, Graph Cut, Glister

## Formulas

These are the formulas used in the LENS method, with detailed explanations of each variable and component.

1. **In-context Prediction Formula**

The prediction for a test input is computed as:

$$
\arg\max_{y \in Y} p_{G}\Bigl(y \mid x_1 \oplus y_1 \cdots x_n \oplus y_n \oplus x_{test}\Bigr),
$$

where:

- $G$ represents the language model used for in-context learning.
- $\{x_i, y_i\}_{i=1}^{n}$ is the set of $n$ in-context examples, each consisting of an input $x_i$ and its corresponding label or output $y_i$.
- $x_{test}$ is the test input for which we want to generate a prediction.
- $Y$ denotes the label space over which the prediction is made.
- $\oplus$ represents the concatenation operation.
- $p_{G}(y \mid \cdot)$ is the probability estimated by the language model for generating output $y$ given the concatenated prompt.
- $\arg\max_{y \in Y}$ means that the overall prediction is the label $y$ that maximizes the conditional probability.

2. **InfoScore Formula**

The InfoScore quantifies the individual in-context informativeness of a single example $e = \{x, y\}$ using feedback from the language model:

$$
I(e, D) = \sum_{e' \in D} c(e, e'),
$$

where:

- $e = \{x, y\}$ is the example, composed of input $x$ and label $y$.
- $D$ is the training dataset over which the informativeness score is aggregated.
- $e' = \{x', y'\}$ is an element of the training dataset.
- $c(e, e') = p_{G}(y' \mid x, y, x') - p_{G}(y' \mid x')$ measures the contribution gap of $e$ when predicting $y'$.
- $I(e, D)$ reflects the overall informativeness of the example $e$ across the training set.

3. **Diversity-Guided Example Update Formula**

This formula updates a chosen example by considering its informativeness and diversity relative to other examples:

$$
e^*_{new} = \arg\max_{e \in D'} s(e, E' - e^*),
$$

with the scoring function:

$$
s(e, E') = I(e, S) - \lambda \sum_{e' \in E'} \text{sim}\bigl(f(e), f(e')\bigr).
$$

Details of the components:

- $e^*$ is the previously chosen example being updated.
- $e^*_{new}$ is the updated example from subset $D'$.
- $f(e) = \Bigl[c(e, e^s_1),\; c(e, e^s_2),\; \ldots,\; c(e, e^s_{|S|})\Bigr]$ is the feature vector of the example $e$.

**Summary**:
- The first formula selects the output $y$ for the test input that maximizes the likelihood using in-context examples.
- The InfoScore formula measures how much a given example improves prediction probabilities over a dataset.
- The Diversity-Guided Example Update Formula combines informativeness and diversity to select the best candidate.

## Training Flow

### Training Steps

1. **Select Dataset and Define Task**: Select a dataset for in-context learning, involving the selection of "support examples."
2. **Filter-Then-Search Method (LENS)**:
   - **Stage 1: Informative Examples Filtering**: 
     - Compute InfoScore for each example.
     - Use progressive filtering to reduce uninformative examples.
   - **Stage 2: Diversity-Guided Example Search**: 
     - Initialize permutations of selected examples.
     - Update by diversity-guided search and evaluate permutations.

#### Sample Pseudocode

```python
# Initialize dataset and language model
dataset = load_dataset(D)
language_model = GPT2()

# Filter stage
filtered_examples = []
score_set = random.sample(dataset, initial_size)

while len(dataset) > target_size:
    infoscores = calculate_infoscore(dataset, score_set)
    dataset = filter_examples(dataset, infoscores)
    score_set = update_score_set(score_set, dataset)

# Search stage
candidate_permutations = initialize_permutations(filtered_examples)
for iteration in range(max_iterations):
    for permutation in candidate_permutations:
        updated_permutation = diversity_guided_update(permutation)
        evaluate_on_validation(updated_permutation)

# Select top-performing permutations as support examples
support_examples = select_top_permutations(candidate_permutations)
```

The method emphasizes filtering for informativeness and searching for diversity to represent the task thoroughly.

## Inference Flow

### Inference Steps

1. Define the task and setup the language model.
2. Construct input by concatenating examples with the task input.
3. Compute prediction using the language model.
4. Measure informativeness using InfoScore.
5. Implement filtering to reduce computational cost.
6. Initialize and refine diverse example permutations.

#### Inference Pseudocode

```python
def in_context_inference(language_model, dataset, task_input):
    # Concatenate examples and task input
    input_sequence = concatenate_examples_and_input(dataset, task_input)
    # Compute model output
    prediction = language_model.predict(input_sequence)
    return prediction

def main():
    # Initialize language model
    language_model = LanguageModel()
    # Load dataset and task input
    dataset, task_input = load_data()
    # Filter Informative Examples
    candidates = progressive_filtering(dataset, initial_set_size=10, progressive_factor=2, m_size=500)
    # Diversity-guided search
    best_permutation = diversity_guided_search(candidates, validation_set, diversity_weight=1, beam_size=8, iterations=10)
    # Make prediction using best permutation
    prediction = in_context_inference(language_model, best_permutation, task_input)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
```

## Experiments

### List of Experiments

* Impact of progressive filtering and candidate selection (Table 5)
* Main performance comparison across various datasets (Table 2)
* Order sensitivity of support examples vs. random examples (Figure 2)
* Transferability of support examples across language models (Table 3)
* Influence of ground truth labels on performance in ICL (Figure 3)
* Impact of hyper-parameters (Table 4)

## Proofs

### Explanation

The paper relies on empirical results rather than traditional proofs. The methodology is justified through experiments and empirical evidence, focusing on the algorithmic complexity.

- **The Complexity of Our Method**: The progressive filtering stage complexity is $O(N \cdot \log_\rho N)$, where $N$ is the training set size. Further details on complexity are discussed under "The Complexity of Our Method."

The paper emphasizes empirical validation over mathematical proofs, providing algorithmic explanations and focusing on experimental results compared to baselines.