# Scalable Influence and Fact Tracing for Large Language Model Pretraining

## Meta

* **Name:** Scalable Influence and Fact Tracing for Large Language Model Pretraining
* **Journal:** International Conference on Learning Representations (ICLR)
* **Year:** 2025
* **Authors:** Google DeepMind, UC San Diego
* **Code:** [GitHub link](https://github.com/pair-code/pretraining-tda)
* **One-liner:** The paper refines gradient-based methods for scalable influence and fact tracing in large language model pretraining without subsampling or pre-filtering.
* **Model:** 154M, 1B, and 8B parameter language models
* **Datasets:** English C4, T-REx, KILT
* **Baselines:** BM25, Gecko embeddings, TRAK, various ablated versions of TrackStar

## Formulas

Below is a breakdown of each formula and the variables used in the TrackStar gradient-based influence method, with explanations in **MathJax** style LaTeX.

### 1. Influence between Two Examples

The influence from a training example $ z_m $ to an evaluation (query) example $ z_q $ is computed as:

$$
I_\theta(z_m, z_q) = \bar{G}_\theta(z_m) \cdot \bar{G}_\theta(z_q)
$$

where:

- $ I_\theta(z_m, z_q) $: The influence score between the training example $ z_m $ and the evaluation (query) example $ z_q $ given the model parameters $ \theta $. A higher value indicates a stronger influence of one example on the other.
- $ \bar{G}_\theta(z_m) $ and $ \bar{G}_\theta(z_q) $: These are the **projected, Hessian-corrected, and unit-normalized gradients** for examples $ z_m $ and $ z_q $, respectively.
- The dot product (denoted by "$\cdot$") between these two vectors measures the alignment between the directions of their gradients. The intuition is that if both gradients point in a similar direction, then the training example $ z_m $ is more likely to significantly influence the evaluation example $ z_q $.

### 2. Projected Gradient Correction

The method first computes an intermediate gradient, then corrects and projects it. The equations are as follows:

$$
\bar{G}_\theta(z) = \frac{G_\theta(z)}{||G_\theta(z)||_2}
$$
$$
G_\theta(z) = R^{-1/2} \, Pd \, \frac{\nabla_\theta \text{Loss}(z, \theta)}{\sqrt{V}}
$$

Here:

- $ \bar{G}_\theta(z) $: The normalized gradient for example $ z $ after projection and Hessian-based correction. The normalization is done with respect to the $\ell_2$-norm, which ensures that the gradient vector has unit length.
- $ G_\theta(z) $: The corrected and projected gradient vector for example $ z $ before normalization.
- $ \|G_\theta(z)\|_2 $: The Euclidean ($\ell_2$) norm of $ G_\theta(z) $. Dividing by this norm ensures the vector is normalized.
- $ \nabla_\theta \text{Loss}(z, \theta) $: The gradient of the loss function with respect to the model parameters $ \theta $ for example $ z $. This gradient indicates how the loss changes as the parameters vary.
- $ V $: A second-moment estimate for the gradients, a vector in $\mathbb{R}^{|\theta|}$. This estimate is used to correct high-magnitude components, making the influence measure more stable.
- $ Pd $: A random projection matrix of dimensions $ \mathbb{R}^{d \times |\theta|} $. This projects the high-dimensional gradient into a lower $ d $-dimensional space, thereby reducing computational complexity.
- $ R $: A Hessian (or second-order) approximation matrix of dimensions $ \mathbb{R}^{d \times d} $. It is computed using a Gauss-Newton approximation over the training or evaluation data.
- $ R^{-1/2} $: The inverse square root of $ R $, which acts as a correction factor that approximates an influence similar to a second-order correction. It adjusts the gradients by incorporating curvature information.

### 3. Correction and Normalization Terms

Additional details about some of the operations and variables:

- **Loss Gradient:**
  
  $$
  \nabla_\theta \text{Loss}(z, \theta)
  $$
  
  This is the standard gradient of the loss function for an example $ z $ with respect to the model parameters $ \theta $.

- **Second Moment Estimate:**
  
  $$
  V \in \mathbb{R}^{|\theta|}
  $$
  
  Here, $ V $ represents an element-wise moving average of the squared gradients. It helps in mitigating issues with gradient components that have high variance.

- **Random Projection Matrix:**
  
  $$
  Pd \in \mathbb{R}^{d \times |\theta|}
  $$
  
  The matrix $ Pd $ is used to reduce the high-dimensional parameter space into a lower-dimensional space ($ d $ dimensions) to maintain computational efficiency while preserving the essential structure of the gradient.

- **Hessian Approximation Matrix:**
  
  $$
  R \in \mathbb{R}^{d \times d}
  $$
  
  In this context, $ R $ is computed using a combination of evaluations over a held-out dataset (denoted $ R_{\text{eval}} $) and the training data (denoted $ R_{\text{train}} $):

  $$
  R = \lambda R_{\text{eval}} + (1 - \lambda) R_{\text{train}}
  $$
  
  where:
  
  - $ R_{\text{eval}} $: The Hessian approximation computed using evaluation data.
  - $ R_{\text{train}} $: The Hessian approximation computed using training data.
  - $ \lambda $: A mixing hyperparameter (with $ 0 \leq \lambda \leq 1 $) that balances the contribution between evaluation and training data in constructing $ R $.

### 4. Summary of the Process

- The raw gradient $ \nabla_\theta \text{Loss}(z, \theta) $ is first computed for a given example $ z $.
- To stabilize the gradient, a per-coordinate division by $ \sqrt{V} $ is performed.
- The resulting vector is then projected into a lower-dimensional space using the random projection matrix $ Pd $.
- Next, a correction factor $ R^{-1/2} $ (obtained from a Gauss-Newton or Hessian approximation) adjusts the gradient to account for curvature, yielding $ G_\theta(z) $.
- Finally, the vector $ G_\theta(z) $ is normalized (using its $\ell_2$-norm) to produce $ \bar{G}_\theta(z) $, which is used to compare influences via the dot product.

This method enhances both interpretability and computational scalability for analyzing influence in large language model pretraining by carefully stitching together gradient corrections, normalization, and dimensionality reduction.

## Training Flow

### Training Flow

1. Begin with a training dataset consisting of a pretraining corpus containing over 160B tokens, aiming to specifically trace impactful examples for an 8B-parameter language model.
2. Use a gradient-based influence method, TrackStar, which follows this formula:
    
   $$
   I_\theta(z_m, z_q) = \bar{G}_\theta(z_m) \cdot \bar{G}_\theta(z_q)
   $$

   where $ \bar{G}_\theta(z) $ is the projected, Hessian-corrected, and unit-normalized gradient, computed as:
   
   $$
   \bar{G}_\theta(z) = \frac{G_\theta(z)}{\|G_\theta(z)\|_2}
   $$

   $$
   G_\theta(z) = R^{-1/2}P_d \frac{\nabla_\theta \text{Loss}(z, \theta)}{\sqrt{V}}
   $$

   Here, $ R $ is a task-specific Hessian approximation, $ P_d $ is a random projection matrix, and $ \sqrt{V} $ is a second moment estimate.
3. For each training example $ z_m $ and a query example $ z_q $, compute the influence score using the gradients of their respective model losses, leveraged with task-specific corrections based on previous analyses.
4. Identify proponents, which are the examples from the training data having the highest influence scores on given queries.
5. Quantify and evaluate the effect of these proponents using the influence metric - tail-patch, which involves calculating the increase in the probability of a target sequence after a single training step using a top-k set of proponents.
6. Conduct evaluations using a mixture of traditional fact tracing metrics (MRR and recall) alongside the influence impact metric.
7. Balance the correct Hessian approximation and optimizer corrections to refine this influence calculation for better result extraction from large datasets without needing pre-filtering techniques.

## Inference Flow

### Inference Flow

1. Compute the gradient dot product of the projected and Hessian-corrected model gradients for each pair of query and training example.
2. Apply optimizer state correction using the second moment estimate of each gradient component to address high-magnitude outliers.
3. Normalize gradients via unit norming to reduce the influence of outlier training examples.
4. Use random projection to decrease gradient dimensionality and ensure storage efficiency.
5. Approximate the Hessian matrix using a mix of training and evaluation gradients to focus on task-specific components.
6. Calculate the cosine similarity between normalized, corrected gradients of each training and query example pair to determine influence.
7. Retrieve top proponents based on their influence score for each query.
8. Evaluate influence through the concept of tail-patching by applying incremental training and observing changes in the prediction probability.

### Inference Flow Code

```python
import torch
from torch.nn import functional as F

def compute_influence(query_gradients, train_gradients, R_train, R_eval, V):
    # Normalize the gradients
    def normalize(grad):
        return grad / torch.norm(grad, p=2)

    # Hessian correction: compute mixed Hessian matrix
    lambda_mixing = 0.99
    R = lambda_mixing * R_eval + (1 - lambda_mixing) * R_train
    R_half_inv = torch.linalg.inv(torch.sqrt(R))

    # Preprocess gradients
    def preprocess_gradient(grad):
        grad_corrected = (grad / torch.sqrt(V)) @ R_half_inv
        return normalize(grad_corrected)
    
    query_grad_processed = preprocess_gradient(query_gradients)

    # Compute cosine similarities between query gradients and train gradients
    influences = []
    for train_grad in train_gradients:
        train_grad_processed = preprocess_gradient(train_grad)
        influence_score = F.cosine_similarity(query_grad_processed, train_grad_processed, dim=0)
        influences.append(influence_score)

    return sorted(range(len(influences)), key=lambda i: influences[i], reverse=True)

# Holds query gradients and training gradients
query_gradients = ...
train_gradients = ...
R_train = ...
R_eval = ...
V = ...

# Compute influence score and retrieve top k influential examples
top_k_indices = compute_influence(query_gradients, train_gradients, R_train, R_eval, V)
top_k_proponents = [train_gradients[i] for i in top_k_indices[:k]]
```

## Experiments

### List of Experiments

* **T-REx Closed Set Evaluation (Table 1)**
  - MRR and Recall@10 Results for the TrackStar method and other baselines
  - Tail-Patch results: Incremental training probability increase (influence metric)
  - Detailed ablation studies of TrackStar configurations
* **C4 Open Set Evaluation (Table 2)**
  - MRR and Recall@10 Results for proponents retrieved from C4
  - Tail-Patch results for different methods
* **Examination of Headroom Analysis (Section 7)**
  - Detailed categorization and analysis of proponents retrieved for correctly vs. incorrectly predicted facts (Figure 3 and Table A.1)
* **Projection Dimensionality Ablation (Figure 2)**
  - Examination of MRR and Tail-Patch scores as a function of gradient projection dimensionality
* **Model Scaling and Influence (Figure 2, right)**
  - Analysis of the correlation between model improvements and attribution scores throughout training
* **Method Output Function Ablations (Table A.3)**
  - Comparison of alternative output functions used in TrackStar: Loss, Margin, and Logit
* **Tail-Patch Results for Top-k Proponents (Table A.6)**
  - Tail-Patch scores averaged over the top 1, 3, 5, and 10 proponents for different models
* **Results Split by Model Correctness (Table A.7)**
  - MRR and Tail-Patch scores differentiated by whether the model prediction was correct or incorrect

Each experiment investigates different aspects of the method's effectiveness in both retrieving influential training examples and understanding how examples affect predictions across language models.

## Proofs

### List of Proofs

* **Submodularity of Influence Function:** The paper discusses gradient-based influence methods which rely on parametric approximations to predict how a model’s behavior would change under the removal (or addition) of specific training examples. The theoretical underpinnings of these methods often rely on demonstrating properties such as submodularity to justify efficient optimization and retrieval.

* **Lower Bound Termination Guarantee:** The proposed method, TrackStar, and its evaluation include time complexity considerations that implicitly involve lower bound guarantees. Particularly, in large-scale retrieval settings, ensuring that the method terminates within a reasonable time frame while processing massive datasets is crucial. This involves theoretical bounds on the efficiency and effectiveness of retrieval operations from vast corpora.

* **Time Complexity Analysis:** Throughout the paper, there is an analysis of the computational cost associated with the proposed influence function estimation, emphasizing scalability with respect to the model and dataset size. This involves assessing both theoretical and empirical runtime complexities as the approach scales to LLMs up to 8B parameters and corpora up to 160B tokens without subsampling or pre-filtering. The time complexity is analyzed in terms of the model parameters and the potential optimizations employed (e.g., random projection, approximations).