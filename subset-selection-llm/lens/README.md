# Leveraging Support Examples for Enhanced In-Context Learning

## Meta

* **Name**: Finding Support Examples for In-Context Learning
* **Authors**: School of Computer Science, Fudan University; Shanghai Key Laboratory of Intelligent Information Processing, Fudan University
* **Code**: [GitHub Repository](https://github.com/LeeSureman/ICL_Support_Example)
* **One-Liner**: This paper introduces the LENS method for identifying "support examples" that enhance in-context learning by selecting informative and representative instances from a dataset.
* **Model Utilized**: GPT-2 (specifically GPT2-L)
* **Datasets Used**: SST-2, SST-5, Amazon, MR, Subj, TREC, AGNews, DBPedia
* **Baselines Compared**: Zero-shot, Random, Random & Validation, Herding, K-Center Greedy, Entropy, Least Confidence, Margin, Cal, Forgetting, GraNd, CRAIG, GradMatch, Facility Location, Graph Cut, Glister

## Formulas

The section below provides a detailed breakdown of critical formulas using MathJax-style LaTeX notation.

### Prediction in In-Context Learning

The prediction formula in the context of in-context learning is given by:

\[
\hat{y} = \underset{y \in Y}{\arg\max}\; p_G\Big(y \mid x_1 \oplus y_1 \oplus \cdots \oplus x_n \oplus y_n \oplus x_{\text{test}}\Big),
\]

**Variables Explained:**

- \(\hat{y}\): Predicted label for the test input.
- \(Y\): Set of all possible labels.
- \(p_G(\cdot \mid \cdot)\): Conditional probability computed by the language model \(G\).
- \(x_1, x_2, \ldots, x_n\): Input features of the \(n\) training examples.
- \(y_1, y_2, \ldots, y_n\): Labels corresponding to each input feature.
- \(x_{\text{test}}\): The test input.
- \(\oplus\): Denotes concatenation.

This formula illustrates how the language model predicts an output by considering a concatenated context of examples and the test input.

### Measuring Informative Examples with InfoScore

Two main formulas help quantify informativeness:

1. **Overall Informativeness**:

\[
I(e, D) = \sum_{e' \in D} c(e, e').
\]

2. **Contribution to Another Example**:

\[
c(e, e') = p_G\big(y' \mid x, y, x'\big) - p_G\big(y' \mid x'\big).
\]

**Variables Explained:**

- \(e\): An example \(\{x, y\}\).
- \(D\): Dataset comprising other examples.
- \(e'\): An example in \(D\), represented as \(\{x', y'\}\).
- The probabilities \(p_G\) measure label prediction based on context inclusion/exclusion.

These formulas aggregate how individual examples contribute to predictions, thus highlighting informativeness.

### Diversity-Guided Example Search

1. **Selecting a New Candidate**:

\[
e^*_{\text{new}} = \underset{e \in D'}{\arg\max}\; s(e, E'),
\]

**Variables Explained:**

- \(D'\): Pool of candidate examples.
- \(E'\): Current sequence of examples.
- \(s(e, E')\): The score function combining informativeness and diversity.

2. **Scoring Function**:

\[
s(e, E') = I(e, S) - \lambda \sum_{e' \in E'} \text{sim}\Big(f(e), f(e')\Big),
\]

**Variables Explained:**

- \(I(e, S)\): Informativeness score.
- \(\lambda\): Weighs the balance between informativeness and diversity.
- \(\text{sim}(\cdot, \cdot)\): Similarity measure, e.g., cosine similarity.

3. **Feature Vector Construction**:

\[
f(e) = \Big[c\big(e, e_s^1\big), c\big(e, e_s^2\big), \ldots, c\big(e, e_s^{|S|}\big)\Big],
\]

**Variables Explained:**

- Captures the contribution of \(e\) across a set of example comparisons.

This set of formulas underlines selecting diverse yet informative examples to improve language model predictions.

## Training Flow

1. **Dataset Selection and Task Definition**: Choose a dataset \(D\) appropriate for in-context learning, aiming to select informative support examples.

2. **Filter-Then-Search (LENS) Method**:
   - **Stage 1: Informative Examples Filtering**
     - Compute InfoScore for each example.
     - Filter out uninformative examples progressively, reducing \(D\) to \(D^\prime\).

   - **Stage 2: Diversity-Guided Example Search**
     - Initiate and update permutations, selecting \(m\) examples.
     - Adjust permutations using a diversity-guided approach, maximizing informativeness and diversity.

```python
# Pseudocode for Training Flow

dataset = load_dataset(D)
language_model = GPT2()

# Filtering stage
filtered_examples = progressive_filtering(dataset, target_size)

# Search stage
best_permutation = diversity_guided_search(filtered_examples, max_iterations)

# Select final support examples
support_examples = select_best_permutations(best_permutation)
```

## Inference Flow

1. **Set Up Task and Language Model**: Define the task with the support of the language model \(G\).

2. **Construct Input**: Concatenate \(n\) examples with the task input.

3. **Model Prediction**:
   - Evaluate predictions using formula (1).
   - Apply InfoScore for example weighting.

4. **Filter and Refine Examples**: Employ progressive filtering and diversity-guided search to refine examples.

5. **Output Predictions**: Use refined examples to complete the task.

```python
# High-Level Pseudocode for Inference

def in_context_inference(language_model, dataset, task_input):
    input_sequence = concatenate_input(dataset, task_input)
    prediction = language_model.predict(input_sequence)
    return prediction

def main():
    dataset, task_input = load_data()
    candidates = progressive_filtering(dataset, 500)
    best_permutation = diversity_guided_search(candidates, 10)
    prediction = in_context_inference(language_model, best_permutation, task_input)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
```

## Experiments

1. **Impact of Progressive Filtering and Selection**: Table 5
2. **Performance Comparison Across Datasets**: Table 2
3. **Order Sensitivity**: Figure 2
4. **Transferability Across Language Models**: Table 3
5. **Influence of Ground Truth Labels**: Figure 3
6. **Impact of Hyper-Parameters**: Table 4

## Proofs

The paper does not provide formal mathematical proofs but rather uses empirical results to support the methodologies used. Notably, the complexity of their proposed method is discussed in terms of its combinatorial and algorithmic nature:

- **Complexity of the LENS Method**: Detailed as \(O(N \cdot \log_\rho N)\).

These provide empirical validations rather than traditional proofs.