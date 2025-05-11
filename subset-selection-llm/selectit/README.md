# SelectIT: Selective Instruction Tuning for LLMs via Uncertainty-Aware Self-Reflection

## Meta

- **Name**: SelectIT: Selective Instruction Tuning for LLMs via Uncertainty-Aware Self-Reflection
- **Journal**: Neural Information Processing Systems (NeurIPS)
- **Year**: 2024
- **Author**: Institute of Computing and Intelligence, Harbin Institute of Technology, Shenzhen, China; NLP2CT Lab, Department of Computer and Information Science, University of Macau
- **Code**: [Github link](https://github.com/Blue-Raincoat/SelectIT)
- **One Liner**: SelectIT uses the intrinsic uncertainty of large language models to select high-quality instruction tuning data without the need for additional resources.
- **Model**: LLaMA-2 (7B, 13B, 70B)
- **Datasets**: Alpaca-GPT4, Selective Alpaca
- **Baselines**: Alpaca-GPT4, LIMA, AlpaGasus, Q2Q, Instruction Mining

## Equations

Below is a detailed breakdown of each equation and its respective variables using MathJax-style LaTeX.

1. **Equation (1): Quality score from LLM**

   $$\text{Quality} \propto \text{Score} \in [1, K] = M(RP, S)$$

   **Explanation of Variables:**

   - $M$: Represents the foundational language model (LLM) used to assess the quality of the instruction tuning (IT) data.
   - $RP$: Denotes the rating prompt, designed to elicit a qualitative score from the model regarding the IT sample.
   - $S$: The sample being evaluated, including both the input $X$ and the response $Y$.
   - $\text{Score} \in [1, K]$: The quality score assigned by the model, in the discrete range from 1 to $K$.

2. **Equation (2): Token-level self-reflection**

   $$S_{base} = \arg\max_{k \in \{1,\dots,K\}} k, \quad P'_{k} = \frac{P_{k}}{\sum_{j=1}^{K}P_{j}}$$

   **Explanation of Variables:**

   - $S_{base}$: Base token-level score, determined by selecting the token $k$ that maximizes the probability.
   - $k \in \{1, \dots, K\}$: Indices of the tokens considered, with $K$ total tokens.
   - $P_{k}$: Raw probability assigned to token $k$.
   - $P'_{k}$: Normalized probability of token $k$.

3. **Equation (3): Token-level score**

   $$S_{\text{token}} = S_{base} \times \frac{1}{K - 1} \sum_{i=1}^{K} |P'_{i} - P'_{S_{base}} |$$

   **Explanation of Variables:**

   - $S_{\text{token}}$: Computed token-level score reflecting internal uncertainty.
   - $S_{base}$: Baseline token score from Equation (2).
   - $P'_{i}$, $P'_{S_{base}}$: Normalized probabilities for tokens.

4. **Equation (4): Sentence-level self-reflection**

   $$S_{\text{sent}} = \frac{\text{Avg}\{S_{\text{token}, i}\}_{i=1}^{K}}{1 + \alpha \times \text{Std}\{S_{\text{token}, i}\}_{i=1}^{K}}$$

   **Explanation of Variables:**

   - $S_{\text{sent}}$: Sentence-level score aggregating token-level scores.
   - $S_{\text{token}, i}$: Token-level score for the $i$-th rating prompt.
   - $\alpha$: Hyperparameter scaling effect of standard deviation.

5. **Equation (5): Model-level self-reflection**

   $$\text{Quality} \propto S_{\text{model}} = \sum_{i=1}^{N} \left(\frac{\theta_{i}}{\sum_{j=1}^{N} \theta_{j}}\right) \times S_{\text{sent}, i}$$

   **Explanation of Variables:**

   - $S_{\text{model}}$: Model-level score offering comprehensive IT data evaluation.
   - $N$: Total number of foundation models.
   - $\theta_{i}$: Parameter count of the $i$-th foundation model.
   - $S_{\text{sent}, i}$: Sentence-level score from the $i$-th model.

In summary, the SelectIT method assesses IT data quality by incorporating self-reflection at multiple levels: token, sentence, and model levels, each assessing different grains of uncertainty and agreement.

## Training Flow

The training flow for SelectIT involves defining the evaluation process, implementing self-reflections on different levels, and selecting high-quality data samples. 

### Training Flow Steps

1. **Define the evaluation process**:
   - Use LLM to rate IT data samples based on internal uncertainty.

2. **Implement Token-level Self-Reflection**:
   - For a sample $S = (X, Y)$, calculate token probabilities and the token-level score $S_{token}$. 

3. **Implement Sentence-level Self-Reflection**:
   - Use multiple rating prompts to compute $S_{sent}$, the sentence-level score considering uncertainty.

4. **Implement Model-level Self-Reflection**:
   - Utilize multiple foundation models to compute a comprehensive model-level score $S_{model}$.

5. **Data selection**:
   - Rank and select top samples as high-quality based on $S_{model}$.

6. **Apply SelectIT**:
   - Evaluate and curate the Selective Alpaca dataset for superior IT dataset construction.

### Training Flow Code

```python
def select_it_data(samples, foundation_models, rating_prompts, alpha):
    def token_reflection(sample, rating_prompt):
        # Compute next-token probabilities
        probs = model.compute_probabilities(sample, rating_prompt)
        S_base = max(prob.index for prob in probs)
        # Token-level reflection
        token_score = S_base * (1 / (len(probs) - 1)) * sum([abs(prob - probs[S_base]) for prob in probs])
        return token_score
    
    def sentence_reflection(sample):
        scores = [token_reflection(sample, rp) for rp in rating_prompts]
        avg_score = mean(scores)
        uncertainty = std(scores)
        return avg_score / (1 + alpha * uncertainty)
    
    def model_reflection(sample):
        sent_scores = [sentence_reflection(sample, model) for model in foundation_models]
        weighted_score = sum([(model.param_count / total_params) * score for model, score in zip(foundation_models, sent_scores)])
        return weighted_score
    
    # Rank samples
    ranked_samples = sorted(samples, key=lambda x: model_reflection(x), reverse=True)
    return ranked_samples[:int(len(samples) * QUALITY_THRESHOLD)]

# Example usage
selected_data = select_it_data(dataset_samples, foundation_models=llama_models, rating_prompts=rating_prompts, alpha=0.2)
```

## Inference Flow

The inference flow involves using the foundation model to compute, rank, and select IT data based on comprehensive assessments.

### Inference Flow Steps

1. Load foundation language model ($M$).
2. Prepare IT dataset $D$ with samples $S = (X, Y)$.
3. Compute token-level, sentence-level, and model-level scores for each sample.
4. Generate final sample quality score $S_{model}$.
5. Select top samples as high-quality IT data.

### Inference Flow Code

```python
def compute_token_score_probabilities(model, input_data, rating_prompts, K):
    # Compute next-token probabilities for each prompt
    token_scores = []
    for prompt in rating_prompts:
        logits = model(input_data + prompt)
        probabilities = F.softmax(logits, dim=-1)
        token_scores.append(probabilities[:, :K].mean(dim=0))
    return torch.stack(token_scores)

def compute_token_reflection_score(token_scores, K):
    disparities = torch.abs(token_scores - token_scores.max(dim=1, keepdim=True)[0])
    token_reflection = token_scores.max(dim=1)[0] * (disparities.sum(dim=1) / (K - 1))
    return token_reflection

def compute_sentence_reflection_score(token_reflection_scores, alpha):
    avg_score = token_reflection_scores.mean()
    std_score = token_reflection_scores.std()
    sentence_reflection = avg_score / (1 + alpha * std_score)
    return sentence_reflection

def compute_model_reflection_score(models, sentence_reflection_scores, model_param_counts):
    weights = torch.tensor(model_param_counts) / sum(model_param_counts)
    model_reflection = (weights * sentence_reflection_scores).sum()
    return model_reflection

def select_high_quality_data(models, data, rating_prompts, alpha, K, top_percentage):
    all_model_scores = []
    for sample in data:
        token_scores = compute_token_score_probabilities(models[0], sample['input'], rating_prompts, K)
        token_reflection_scores = compute_token_reflection_score(token_scores, K)
        sentence_reflection = compute_sentence_reflection_score(token_reflection_scores, alpha)
        
        model_scores = []
        for model in models:
            model_score = compute_model_reflection_score(model, sentence_reflection, model.param_count)
            model_scores.append(model_score)
        all_model_scores.append(sum(model_scores) / len(model_scores))
    
    sorted_indices = sorted(range(len(all_model_scores)), key=lambda i: all_model_scores[i], reverse=True)
    selected_samples = [data[i] for i in sorted_indices[:int(len(data) * top_percentage)]]
    return selected_samples
```

## Experiments

### List of Experiments

1. **Annotation Budget Ablations**: Impact of varying annotation budgets on Selective Alpaca.
2. **Efficiency of SelectIT**: Comparing time cost and efficiency of SelectIT against others.
3. **Performance Across Scales**: Robustness of SelectIT on different scales of foundation models.
4. **Domain-Specific Task Evaluation**: Improving machine translation with SelectIT on ALMA.
5. **Different Reflection Strategy Effects**: Role analysis of token, sentence, and model-level self-reflection.
6. **Data Representation Analysis**: Comparing Selective Alpaca with original datasets.
7. **Correlation Study of Sample Characteristics**: Relationship between sample complexity and performance.
8. **Robustness to Instruction Tuning Datasets**: Adaptability of SelectIT on various datasets like WizardLM.
9. **Statistical Significance Testing**: Reliability check of performance improvements.
10. **MT LLMs Impact Study**: Impact analysis of SelectIT on multilingual translation effectiveness.

## Proofs

### List of Proofs

The paper does not provide explicit proofs for mathematical theorems, but details the methodologies and processes behind its approach. Consequently, there is no formal list of proofs in this work.

