# SelectIT: Selective Instruction Tuning for LLMs via Uncertainty-Aware Self-Reflection

## Meta

- **Name**: SelectIT
- **Journal**: Neural Information Processing Systems (NeurIPS)
- **Year**: 2024
- **Author**: 1Institute of Computing and Intelligence, Harbin Institute of Technology, Shenzhen, China; 2NLP2CT Lab, Department of Computer and Information Science, University of Macau
- **Code**: [Github link](https://github.com/Blue-Raincoat/SelectIT)
- **One liner**: SelectIT uses the intrinsic uncertainty of large language models to select high-quality instruction tuning data without the need for additional resources.
- **Model**: LLaMA-2 (7B, 13B, 70B)
- **Datasets**: Alpaca-GPT4, Selective Alpaca
- **Baselines**: Alpaca-GPT4, LIMA, AlpaGasus, Q2Q, Instruction Mining

## Formulas

Below is a detailed breakdown of each variable in the provided formulas using MathJax-style LaTeX.

### 1. Overall Quality Scoring

The overall goal is expressed as

\[
\text{Quality} \propto \text{Score} \in [1, K] = M(\text{RP}, S)
\]

Here:

- \(\text{Quality}\): A metric indicating the quality of a given sample \(S\). A higher value suggests better IT (instruction tuning) data quality.
- \(\text{Score}\): An evaluation score assigned to the sample \(S\). It lies in the range \([1, K]\) where higher scores represent superior quality.
- \(K\): The maximum rating (or the total number of discrete rating levels). For instance, if \(K = 5\), then the rating scale is from 1 to 5.
- \(M\): The foundation model (or language model) being used to evaluate the sample.
- \(\text{RP}\): The rating prompt provided to the model. This prompt is designed to elicit an evaluative response.
- \(S\): The sample (e.g., an instruction or data point) whose quality is being assessed.

### 2. Token-level Self-Reflection  

Token-level self-reflection is computed in two steps. First, the model computes a base token and the normalized token probabilities:

\[
S_{\text{base}} = \arg \max_{k \in \{1,\ldots,K\}} k, \quad P'_k = \frac{P_k}{\sum_{j=1}^{K} P_j}
\]

- \(S_{\text{base}}\): The base token selected, determined via the argument that maximizes over the rating tokens \(k\). In the simplest interpretation, it can be seen as the token (or rating value) with the highest initial probability.
- \(P_k\): The probability assigned to token (or rating) \(k\) as predicted by the model.
- \(P'_k\): The normalized probability for token \(k\). It is obtained by dividing \(P_k\) by the sum \(\sum_{j=1}^{K} P_j\), so that the probabilities sum to 1 across all \(K\) tokens.

Next, the token-level self-reflection score is defined as:

\[
S_{\text{token}} = S_{\text{base}} \times \frac{1}{K - 1} \times \sum_{i=1}^{K} \Big| P'_i - P'_{S_{\text{base}}} \Big|
\]

- \(S_{\text{token}}\): The token-level self-reflection score, which adjusts the base score by considering the spread (i.e., differences) in the normalized probabilities.
- \(S_{\text{base}}\): Carries forward the initial assessment from the highest probability token.
- \(\frac{1}{K - 1}\): A normalization factor ensuring that the deviation term is appropriately scaled given there are \(K\) rating tokens.
- \(\sum_{i=1}^{K} \Big| P'_i - P'_{S_{\text{base}}} \Big|\): This sum computes the absolute differences between the normalized probability of each token \(P'_i\) and the normalized probability of the base token \(P'_{S_{\text{base}}}\). This term serves as a measure of uncertainty or dispersion in the model’s predictions.

### 3. Sentence-level Self-Reflection  

To aggregate the information from multiple tokens (or rating prompts), the sentence-level self-reflection score is defined as:

\[
S_{\text{sent}} = \frac{\operatorname{Avg}\{S_{\text{token}}^i\}_{i=1}^{K}}{1 + \alpha \times \operatorname{Std}\{S_{\text{token}}^i\}_{i=1}^{K}}
\]

- \(S_{\text{sent}}\): The sentence-level self-reflection score; it provides a refined quality score by combining token-level assessments.
- \(\operatorname{Avg}\{S_{\text{token}}^i\}_{i=1}^{K}\): The average of the token-level scores over the \(K\) rating prompts. Here, \(S_{\text{token}}^i\) indicates the token-level score from the \(i\)-th rating prompt.
- \(\operatorname{Std}\{S_{\text{token}}^i\}_{i=1}^{K}\): The standard deviation of the token-level scores across the \(K\) ratings. This term captures the variability or consistency among the token-level predictions.
- \(\alpha\): A hyperparameter that controls how much weight is given to the variability (measured by the standard deviation). A higher \(\alpha\) increases the penalty for disagreement among token-level predictions.
- The denominator \(1 + \alpha \times \operatorname{Std}\{\cdot\}\) serves to moderate the average token score by the uncertainty present in the token-level outputs.

### 4. Model-level Self-Reflection  

To further refine the evaluation and reduce dependence on a single model, model-level self-reflection leverages multiple foundation models:

\[
S_{\text{model}} = \left( \sum_{i=1}^{N} \frac{\theta_i}{\sum_{j=1}^{N} \theta_j} \, S_{\text{sent}}^i \right)
\]

*(Note: In the provided formula, the multiplication is shown outside the sum. However, the intended interpretation is typically a weighted sum of the sentence-level scores.)*

- \(S_{\text{model}}\): The model-level self-reflection score. This aggregated metric incorporates outputs from several models.
- \(N\): The number of different foundation models used for evaluation.
- \(\theta_i\): The parameter count (or a weight related to the model's size) of the \(i\)-th foundation model. This is used as a proxy for the model’s capability.
- \(\frac{\theta_i}{\sum_{j=1}^{N} \theta_j}\): The normalized weight assigned to the \(i\)-th model. Models with more parameters (and potentially higher capability) receive proportionally more weight.
- \(S_{\text{sent}}^i\): The sentence-level self-reflection score provided by the \(i\)-th model.
- The summation \(\sum_{i=1}^{N} \frac{\theta_i}{\sum_{j=1}^{N} \theta_j} \, S_{\text{sent}}^i\) aggregates the individual sentence-level scores from each model into a single, overall score.

### Summary  

- The overall quality is determined by a foundation model \(M\) that combines the rating prompt \(\text{RP}\) and sample \(S\).
- At the token level, the model’s confidence distribution (through \(P_k\) and \(P'_k\)) informs an uncertainty measure which adjusts a base score \(S_{\text{base}}\) to yield \(S_{\text{token}}\).
- These token-level scores are aggregated at the sentence level, where their mean is balanced against their variability (modulated by \(\alpha\)) to produce \(S_{\text{sent}}\).
- Finally, multiple models’ sentence-level scores are combined using weights based on their parameter counts (\(\theta_i\)) to obtain a robust model-level score \(S_{\text{model}}\), which in turn is proportional to the overall quality assessment.

This approach leverages uncertainty-aware self-reflection at multiple granular levels (token, sentence, and model) to better evaluate and curate high-quality datasets for instruction tuning large language models.

## Training Flow

### Training Flow

1. Define the evaluation process:
   - Utilize the foundation language model (LLM) to determine rating scores for IT data samples based on internal uncertainty in the LLM.
   
2. Implement Token-level Self-Reflection:
   - For a given sample \(S = (X, Y)\) and a rating prompt \(RP\), calculate the next-token probability \(P'_{k}\) via the foundation model for tokens \(k \in \{1, ..., K\}\).
   - Determine the token-level score \(S_{token}\) by adjusting \(S_{base}\), the token with the highest normalized probability: 
   \[
   S_{token} = S_{base} \cdot \frac{1}{K - 1} \sum_{i=1}^K |P'_i - P'_{S_{base}}|
   \]
   
3. Implement Sentence-level Self-Reflection:
   - Define \(K\) semantically similar rating prompts \{RP_0, RP_1, ..., RP_K\}.
   - Compute sentence-level score \(S_{sent}\) by averaging token-level scores from these prompts and integrating the uncertainty:
   \[
   S_{sent} = \frac{\text{Avg}\{S_{token}\}}{1 + \alpha \cdot \text{Std}\{S_{token}\}}
   \]
   
4. Implement Model-level Self-Reflection:
   - For multiple foundation models with parameter counts \(\{\theta_1, \theta_2, ..., \theta_N\}\), compute a comprehensive model-level score \(S_{model}\) by weighing sentence-level scores:
   \[
   S_{model} = \sum_{i=1}^N \left( \frac{\theta_i}{\sum_{j=1}^N \theta_j} \cdot S_{sent}^{i} \right)
   \]

5. Data selection:
   - Use \(S_{model}\) to rank the dataset samples.
   - Select top samples as high-quality IT data.

6. Apply SelectIT on dataset:
   - Use SelectIT to evaluate and curate the Selective Alpaca dataset, resulting in a superior IT dataset for model training.

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

This pseudocode outlines the high-level process of applying SelectIT in Python using PyTorch to evaluate the IT data quality and select a high-quality dataset for training LLMs.

## Inference Flow

### Inference flow

1. Load the foundation language model (\(M\)).
2. Prepare instruction tuning (IT) dataset \( D \), consisting of samples \( S = (input \, X, response \, Y) \).
3. For each sample \( S \) in the IT dataset:
    1. Use token-level uncertainty: For each token \( k \in \{1, \ldots, K\} \), calculate the probability \( P_k \) of the next token. Compute the base score \( S_{base} \) using \( S_{base} = \arg \max_k P'_k \), where \( P'_k = \frac{P_k}{\sum_{j=1}^K P_j} \).
    2. Token-level self-reflection: Enhance \( S_{base} \) to \( S_{token} \) by accounting for the average disparity in token-score probabilities using: 
       \[
       S_{token} = S_{base} \times \frac{1}{K-1} \sum_{i=1}^{K} |P'_i - P'_{S_{base}}|
       \]
    3. Sentence-level uncertainty: Use \( K \) semantically similar rating prompts \{RP0, ..., RPK\} to generate a series of quality scores \{\( S_{token}^0, ..., S_{token}^K \)\}. Compute sentence score \( S_{sent} \) using:
       \[
       S_{sent} = \frac{\text{Avg}\{S_{token}^i\}_{i=1}^K}{1 + \alpha \times \text{Std}\{S_{token}^i\}_{i=1}^K}
       \]
    4. Model-level self-reflection: Aggregate sentence scores across multiple foundation models \( \{M_1, M_2, ..., M_N\} \) based on their parameter counts \(\{\theta_1, ..., \theta_N\}\), computing:
       \[
       S_{model} = \left( \sum_{i=1}^{N} \frac{\theta_i}{\sum_{j=1}^{N} \theta_j} \right) \times S_{sent}^i
       \]
4. Generate final sample quality score \( S_{model} \) and sort all samples in \( D \) by \( S_{model} \).
5. Select the top proportion of samples with the highest \( S_{model} \) scores as high-quality IT data.

### Inference flow Code

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
        
        # Repeat with different models if necessary
        model_scores = []
        for model in models:
            model_score = compute_model_reflection_score(model, sentence_reflection, model.param_count)
            model_scores.append(model_score)
        all_model_scores.append(sum(model_scores) / len(model_scores))  # Average over models
    
    # Select top samples based on model reflection scores
    sorted_indices = sorted(range(len(all_model_scores)), key=lambda i: all_model_scores[i], reverse=True)
    selected_samples = [data[i] for i in sorted_indices[:int(len(data) * top_percentage)]]
    return selected_samples
```

This pseudocode describes the inferencing process of SelectIT, illustrating how it ranks IT data samples based on their estimated quality and how high-quality samples are selected using uncertainty from multiple LLM levels.

## Experiments

### List of Experiments

1. **Annotation Budget Ablations**: Investigating the impact of varying annotation budgets on the performance of Selective Alpaca (Figures and tables discussed in text).

2. **Efficiency of SelectIT**: A comprehensive comparison of time cost and efficiency between SelectIT and other data selection methods, such as ChatGPT and GPT-4 API (Table 9).

3. **Performance Across Scales**: Evaluation of the robustness of SelectIT when applied to various scales of foundation models like LLaMA-2 and LLaMA-3 (Table 6).

4. **Domain-Specific Task Evaluation**: Applying SelectIT to improve machine translation capabilities within domain-specific tasks, examining its effectiveness in enhancing the ALMA model (Table 8).

5. **Different Reflection Strategy Effects**: Analysis of the distinct roles played by token, sentence, and model-level self-reflection in the data selection process of SelectIT (Table 4).

6. **Data Representation Analysis**: Examination of data representation and characteristics comparing Selective Alpaca with original datasets to understand the distribution of selected data points (Figures 4 and 5).

7. **Correlation Study of Sample Characteristics**: Evaluation of the relationship between sample computation complexity, average length, and model performance driving insights into high-quality IT data traits (Figure 6).

8. **Robustness to Instruction Tuning Datasets**: Testing the adaptability of SelectIT by deploying on various widely-utilized instruction tuning datasets like WizardLM and Orca-GPT4 (Table 7).

9. **Statistical Significance Testing**: Application of statistical significance tests to ensure the reliability of performance improvements with SelectIT (discussed in Appendix A.1).

10. **MT LLMs Impact Study**: Impact analysis of SelectIT on Machine Translation large language models, evaluating multilingual translation effectiveness (Table 11).

## Proofs

### List of Proofs

The paper "SelectIT: Selective Instruction Tuning for LLMs via Uncertainty-Aware Self-Reflection" does not explicitly state proofs for mathematical theorems or propositions. However, it discusses the processes and methodologies underlying its approach. As such, the paper does not contain a list of proofs in the traditional mathematical sense.