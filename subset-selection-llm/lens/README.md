# Finding Support Examples for In-Context Learning

## Meta
* **Name**: Finding Support Examples for In-Context Learning
* **Journal**: Not specified
* **Year**: Not specified
* **Author**: School of Computer Science, Fudan University; Shanghai Key Laboratory of Intelligent Information Processing, Fudan University
* **Code**: [GitHub link](https://github.com/LeeSureman/ICL_Support_Example)
* **One-liner**: The paper proposes the LENS method to find "support examples" for in-context learning that improve performance by selecting informative and representative examples from a dataset.
* **Model**: GPT-2 (specifically GPT2-L)
* **Datasets**: SST-2, SST-5, Amazon, MR, Subj, TREC, AGNews, DBPedia
* **Baselines**: Zero-shot, Random, Random & Validation, Herding, K-Center Greedy, Entropy, Least Confidence, Margin, Cal, Forgetting, GraNd, CRAIG, GradMatch, Facility Location, Graph Cut, Glister

## Formulas

### 1. In-Context Prediction Formula

The prediction for a test input is computed as:

$$
\arg\max_{y \in Y} p_{G}\Bigl(y \mid x_1 \oplus y_1 \cdots x_n \oplus y_n \oplus x_{test}\Bigr),
$$

where:

- $G$: Represents the language model (LM) used for in-context learning.
- $\{x_i, y_i\}_{i=1}^{n}$: A set of $n$ in-context examples, consisting of input $x_i$ and label/output $y_i$.
- $x_{test}$: The test input for which we want to generate a prediction.
- $Y$: The label (or output) space over which the prediction is made.
- $\oplus$: Represents the concatenation operation.
- $p_{G}(y \mid \cdot)$: The probability estimated by the language model $G$ for generating output $y$ given the concatenated prompt.
- $\arg\max_{y \in Y}$: Indicates that the overall prediction is the label $y$ that maximizes the conditional probability.

### 2. InfoScore Formula

The InfoScore quantifies the individual in-context informativeness:

$$
I(e, D) = \sum_{e' \in D} c(e, e'),
$$

where

$$
c(e, e') = p_{G}(y' \mid x, y, x') - p_{G}(y' \mid x'),
$$

and:

- $e = \{x, y\}$: An example composed of input $x$ and label $y$.
- $D$: The training dataset over which the informativeness score is aggregated.
- $e' = \{x', y'\}$: An element of the training dataset.
- $p_{G}(y' \mid x, y, x')$: Probability assigned by the language model $G$ to $y'$ with additional context $e$.
- $p_{G}(y' \mid x')$: Probability assigned by the language model to $y'$ when conditioned only on $x'$.
- $c(e, e')$: The gap measuring additional context $e$ contribution.
- $I(e, D)$: Aggregated informativeness across the training set.

### 3. Diversity-Guided Example Update Formula

This formula updates a chosen example by considering informativeness and diversity:

$$
e^*_{new} = \arg\max_{e \in D'} s(e, E' - e^*),
$$

where:

$$
s(e, E') = I(e, S) - \lambda \sum_{e' \in E'} \text{sim}\bigl(f(e), f(e')\bigr).
$$

The components include:

- $e^*$: The previously chosen example.
- $e^*_{new}$: The updated example.
- $D'$: A subset of the training dataset.
- $E' = E - e^*$: A set of examples with $e^*$ removed.
- $s(e, E')$: A composite score reflecting both informativeness and diversity.
- $\lambda$: A hyperparameter controlling the trade-off between informativeness and diversity.
- $\text{sim}\bigl(f(e), f(e')\bigr)$: A similarity measure between feature vectors.
- $f(e)$: Feature vector of example $e$, considering its informativeness with respect to the score set $S$.

## Training Flow

### Training Flow

1. **Select Dataset and Define Task**: Start with a dataset $D$ for in-context learning to select support examples.
2. **Filter-Then-Search Method (LENS)**:
   - **Stage 1: Informative Examples Filtering**
     - Compute InfoScore $I(e, D)$ for each example to evaluate informativeness.
     - Use progressive filtering (Algorithm 1) to reduce $D$ to a smaller subset $D^\prime$.
   - **Stage 2: Diversity-Guided Example Search**
     - Initialize permutations of selected examples.
     - Iteratively update via diversity-guided search (Algorithm 2).
     - Select permutations with superior task performance.

### Sample Pseudocode

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

The method highlights filtering for informativeness and diversity-search for representative examples.

## Inference Flow

### Inference Flow

1. Define the task and setup the language model $G$.
2. Concatenate $n$ examples with the task input.
3. Compute prediction using:

$$
\arg \max_{y \in Y} p_G(y|x_1 \oplus y_1 \cdots x_n \oplus y_n \oplus x_{\text{test}})
$$

4. Measure informativeness with InfoScore.
5. Implement progressive filtering to lessen computational costs.
6. Initialize diverse example permutations and refine them.
7. Update permutations for diversity and informativeness.
8. Evaluate permutations on a validation set.
9. Output the final set of support examples.

### Inference Flow Code

```python
def in_context_inference(language_model, dataset, task_input):
    # Step 1: Concatenate examples and task input
    input_sequence = concatenate_examples_and_input(dataset, task_input)

    # Step 2: Compute model output
    prediction = language_model.predict(input_sequence)

    return prediction

def progressive_filtering(dataset, initial_set_size, progressive_factor, m_size):
    current_dataset = dataset
    score_set = random_sample(initial_set_size)

    while len(current_dataset) > m_size:
        # Compute InfoScore for current dataset
        info_scores = [compute_info_score(e, score_set) for e in current_dataset]
        
        # Filter dataset
        if len(current_dataset) / progressive_factor < m_size:
            break
        else:
            top_candidates = select_top_candidates(current_dataset, info_scores, ratio=1/progressive_factor)
            current_dataset = top_candidates
            new_samples = random_sample(len(score_set) * (progressive_factor - 1))
            score_set.extend(new_samples)

    return current_dataset

def diversity_guided_search(candidate_set, val_set, diversity_weight, beam_size, iterations):
    beam = initialize_diverse_permutations(candidate_set, beam_size)

    for _ in range(iterations):
        new_beam = []
        for candidate in beam:
            for _ in range(beam_size):
                updated = update_candidate(candidate, candidate_set, diversity_weight)
                new_beam.append(updated)

        best_candidates = evaluate_beam(new_beam, val_set)
        beam = best_candidates[:beam_size]

    return select_best_permutation(beam)

# Main Function
def main():
    # Initialize language model
    language_model = LanguageModel()

    # Load dataset and task input
    dataset, task_input = load_data()

    # Step 1: Filter Informative Examples
    candidates = progressive_filtering(dataset, initial_set_size=10, progressive_factor=2, m_size=500)

    # Step 2: Diversity-guided search
    best_permutation = diversity_guided_search(candidates, validation_set, diversity_weight=1, beam_size=8, iterations=10)
    
    # Make prediction using best permutation
    prediction = in_context_inference(language_model, best_permutation, task_input)

    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
```

The pseudocode outlines the inference flow for selecting support examples in in-context learning, focusing on filtering and diversity-guided searching.

## Experiments

### Experiments

* Impact of progressive filtering and candidate selection (Table 5)
* Main performance comparison across datasets (Table 2)
* Order sensitivity of support vs. random examples (Figure 2)
* Transferability across different models (Table 3)
* Influence of labels on performance in ICL (Figure 3)
* Impact of hyper-parameters (Table 4)

## Proofs

### Proofs

The paper primarily offers empirical justifications over formal mathematical proofs.

- **The Complexity of Our Method**: The progressive filtering stage complexity is $O(N \cdot \log_\rho N)$, where $N$ is the training set size.

The approach focuses on algorithmic explanations with experimental validation rather than traditional mathematical proofs.