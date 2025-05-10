# Scalable Influence and Fact Tracing for Large Language Model Pretraining

## Meta

- **Name**: Scalable Influence and Fact Tracing for Large Language Model Pretraining
- **Journal**: International Conference on Learning Representations (ICLR)
- **Year**: 2025
- **Authors**: Google DeepMind, UC San Diego
- **Code**: [Github link](https://github.com/pair-code/pretraining-tda)
- **Abstract**: The paper refines gradient-based methods for scalable influence and fact tracing in large language model pretraining without subsampling or pre-filtering.
- **Model Sizes**: 154M, 1B, and 8B parameter language models
- **Datasets**: English C4, T-REx, KILT
- **Baselines**: BM25, Gecko embeddings, TRAK, various ablated versions of TrackStar

## Formulas

In this section, we delve into the mathematical models used in this research. The notation follows the MathJax-style LaTeX formatting.

### Influence Computation

The formula for influence computation is:
\[
I_\theta(z_m, z_q) = \overline{G}_\theta(z_m) \cdot \overline{G}_\theta(z_q)
\]
where:
- \(\overline{G}_\theta(z) = \frac{G_\theta(z)}{\lVert G_\theta(z) \rVert_2}\) denotes the normalized gradient.

**Variables and Meanings**:
- \(\theta\): Model parameters.
- \(z_m\), \(z_q\): Data examples; \(z_m\) is typically a training instance, \(z_q\) a query instance.
- \(G_\theta(z)\): Gradient of loss w.r.t. model parameters \(\theta\) evaluated at \(z\).
- \(\lVert G_\theta(z) \rVert_2\): \(\ell_2\)-norm of gradient vector \(G_\theta(z)\).
- \(\overline{G}_\theta(z)\): Normalized gradient ensuring unit norm.
- \(I_\theta(z_m, z_q)\): Influence score calculated as the dot product of normalized gradients.

### Loss Gradient

The modified loss gradient formula is:
\[
G_\theta(z) = R^{-1/2} \, P_d \frac{\nabla_\theta \text{Loss}(z, \theta)}{\sqrt{V}}
\]

**Variables and Meanings**:
- \(\nabla_\theta \text{Loss}(z, \theta)\): Gradient of loss function w.r.t \(\theta\) at \(z\).
- \(R^{-1/2}\): Inverse square root of transformation matrix \(R\).
- \(P_d\): Projection or selection matrix.
- \(V\): A scalar or diagonal matrix used for normalization.
- \(G_\theta(z)\): The rescaled and normalized gradient.

### Hessian-Corrected Dot Product

This is given by:
\[
-\nabla L(z_q) \, H^{-1} \, \nabla L(z_m)
\]

**Variables and Meanings**:
- \(\nabla L(z_q)\), \(\nabla L(z_m)\): Gradients at \(z_q\) and \(z_m\).
- \(H\): Hessian matrix of second derivatives.
- \(H^{-1}\): (Pseudo-)inverse of the Hessian.

### Autocorrelation Matrix in TRAK

The formula is:
\[
\tilde{\Phi}^T \tilde{\Phi} \in \mathbb{R}^{|\theta| \times |\theta|}
\]

**Variables and Meanings**:
- \(\tilde{\Phi}\): Matrix representing features or gradients.
- \(\tilde{\Phi}^T \tilde{\Phi}\): Autocorrelation matrix showing inner products among features.
- \(|\theta|\): Dimensionality of parameter space.

### Mixing Approach for Matrix \( R\)

The formula is:
\[
R = \lambda R_{\text{eval}} + (1 - \lambda) R_{\text{train}}
\]

**Variables and Meanings**:
- \(R\): Matrix for gradient scaling/normalizing.
- \(R_{\text{eval}}\): Evaluation data-based matrix.
- \(R_{\text{train}}\): Training data-based matrix.
- \(\lambda\): Mixing coefficient between \(0\) and \(1\).

## Training Flow

1. **Dataset**: Start with a pretraining corpus over 160B tokens for an 8B-parameter model.
2. **Gradient-Based Influence Method**: Use TrackStar:
    \[
    I_\theta(z_m, z_q) = \overline{G}_\theta(z_m) \cdot \overline{G}_\theta(z_q)
    \]
3. **Compute Influence Score**: For training \(z_m\) and query \(z_q\).
4. **Identify Proponents**: Training examples with highest influence.
5. **Quantify Influence**: Use tail-patch metric for incremental training effect.
6. **Evaluation**: Employ MRR and recall along with influence metrics.
7. **Refine Calculations**: Combine Hessian approximation and optimizer corrections.

## Inference Flow

1. **Gradient Dot Product**: For each query and training example pair.
2. **Optimizer State Correction**: Use second moment estimates.
3. **Normalize Gradients**: Unit norm to mitigate outlier influence.
4. **Random Projection**: For dimensionality reduction.
5. **Hessian Approximation**: Mix of training and evaluation gradients.
6. **Calculate Cosine Similarity**: Between gradients to find influence.
7. **Retrieve Top Proponents**: Based on influence score.
8. **Tail-Patching**: Test influence via incremental training effects.

## Experiments

- **T-REx Closed Set**: MRR, Recall@10, Tail-Patch, and ablation studies.
- **C4 Open Set**: MRR, Recall@10, and tail-patch results.
- **Headroom Analysis**: Analysis of proponents for prediction correctness.
- **Projection Dimensionality Ablation**: Scores vs. gradient dimensionality.
- **Model Scaling**: Correlation between model improvements and scores.
- **Method Output Ablations**: Explore different output functions.
- **Tail-Patch Proponents**: Scores for top-k proponents.
- **Results by Model Correctness**: MRR and Tail-Patch for correct/incorrect predictions.

## Proofs

- **Submodularity of Influence Function**: Justification of efficient optimization.
- **Lower Bound Termination Guarantee**: Time complexity considerations for TrackStar.
- **Time Complexity Analysis**: Scalability with model size and dataset for influence function estimation.