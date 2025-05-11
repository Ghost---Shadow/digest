# Representative Demonstration Selection for In-Context Learning

## Meta

- **Name:** Representative Demonstration Selection for In-Context Learning with Two-Stage Determinantal Point Process
- **Journal:** Not mentioned
- **Year:** Likely 2023 based on references
- **Author Affiliations:** 
  - School of Artificial Intelligence, University of Chinese Academy of Sciences
  - The Laboratory of Cognition and Decision Intelligence, Institute of Automation, Chinese Academy of Sciences
  - Meituan, Beijing
  - Harbin Institute of Technology, Weihai
  - Shanghai Artificial Intelligence Laboratory
- **Code:** Not provided
- **One-liner:** The paper introduces a two-stage Determinantal Point Process method to select a representative subset of in-context demonstrations, optimizing for quality and diversity to enhance In-Context Learning.
- **Model:** GPT2-xl (1.5B), GPT-J (6B), GPT-NeoX (20B)
- **Datasets:** SST-2, TREC, CB, AGNews, DBPedia, RTE
- **Baselines:** Random, Cluster, DPP, Least Confidence (LC), Cal

## Formulas

1. **Influence of a Demonstration**

   The influence of a demonstration (or example) \( e \) on another instance \( e_i = (x_i, y_i) \) is defined as:

   $$
   \text{Inf}(e, e_i) = p_{\text{LM}}(y_i \mid e, x_i) - p_{\text{LM}}(y_i \mid x_i)
   $$

   - \( e \): A demonstration provided to the language model (LM) to assist in in-context learning.
   - \( e_i = (x_i, y_i) \): An instance from the score set with input \( x_i \) and correct label \( y_i \).
   - \( p_{\text{LM}}(y_i \mid e, x_i) \): Probability assigned to the label \( y_i \) by the LM when given both the demonstration \( e \) and input \( x_i \).
   - \( p_{\text{LM}}(y_i \mid x_i) \): Probability of the label \( y_i \) given only input \( x_i \).

2. **Quality Metric for a Demonstration**

   The quality of a demonstration \( e \) is quantified as:

   $$
   Q(e) = \frac{\sum_{e_i \in \mathcal{D}_{\text{score}}} \text{Inf}(e, e_i)}{T}
   $$

   - \( \mathcal{D}_{\text{score}} \): The score set used for measuring the quality of demonstrations.
   - \( T \): The number of elements in the score set.

3. **Influence Embedding and Quality Representations**

   For each item \( x_j \) in a semantic set \( \mathcal{D}_{\text{sem}} \):

   - \( x_j \in \mathcal{D}_{\text{sem}} \): A candidate demonstration selected based on semantic diversity.
   - \( I_j \): A \( T \)-dimensional vector representing the influence of \( x_j \) on instances in the score set.
   - Influence representation matrix: 

     $$
     \mathbf{I} \in \mathbb{R}^{N_{\text{sem}} \times T}
     $$

   - Quality representation vector:

     $$
     \mathbf{Q} \in \mathbb{R}^{T}
     $$

4. **Combining Influence and Quality into a PSD Matrix**

   Overall influence representation through a positive semidefinite matrix \( \mathbf{L}_I \):

   $$
   \mathbf{L}_I = \mathbf{Q} \cdot \mathbf{I} \, \mathbf{I}^T \cdot \mathbf{Q}
   $$

## Training Flow

1. **Initialize Training Set and Semantic Representations:**
   - Start with a training set \( D = \{e_1, e_2, \cdots, e_N\} \).
   - Encode instances with sentence-BERT for semantic representations.
   
2. **Stage One: Semantic Diversity Selection using DPP:**
   - Compute a PSD matrix for semantic diversity.
   - Select a candidate subset \( D_{\text{sem}} \) using DPP.

3. **Score Set Selection:**
   - Randomly sample a score set from \( D \setminus D_{\text{sem}} \).

4. **Stage Two: High-Quality and Influence Diversity Selection:**
   - Compute influence and quality scores for instances in \( D_{\text{sem}} \).
   - Form influence representation matrix \( \mathbf{III} \) and quality vector \( \mathbf{QQQ} \).
   - Construct another PSD matrix incorporating quality and influence diversity.
   - Select a final demonstration subset \( D_{\text{dem}} \).

## Inference Flow

1. Extract semantic representations using sentence-BERT and construct a PSD matrix.
2. Apply DPP for selecting \( D_{\text{sem}} \).
3. Randomly sample a score set.
4. Compute influence scores and quality scores.
5. Construct a second PSD matrix.
6. Use DPP to select a demonstration subset that maximizes quality and diversity.

```python
...

# Steps outlined in the code section for processing.
```

## Experiments

1. **Impact of Semantic Diversity and Quality:**
   - Semantic diversity, quality, and influence diversity are showcased.
2. **Main Results:**
   - Comparison with baseline methods.
3. **Performance Comparison:**
   - Comparison with retrieval-based methods.
4. **Factors Impacting Performance:**
   - Examining semantic diversity, instance quality, and influence.
5. **Order Sensitivity and Transferability:**
   - Effects shown through figures and tables.
   
## Proofs

- **List of Proofs:**
  - Submodularity of influence function
  - Lower bound termination guarantee
  - Time complexity analysis