# [Differentiable Semantic Metric Approximation in Probabilistic Embedding for Cross-Modal Retrieval](https://proceedings.neurips.cc/paper_files/paper/2022/file/4e786a87e7ae249de2b1aeaf5d8fde82-Paper-Conference.pdf)

**Source Code:** [Github](https://github.com/VL-Group/2022-NeurIPS-DAA)

**Datasets:** MS-COCO, CUB Captions, and Flickr30K

**Author:** Tencent

**Journal:** Neurips

**Year of Submission:** 2022

## What problem does it solve?

ClipTextVisionEmbeddings but many to many

## How does it solve it?

### Training flow

### Inference

### Equations

### Model

## How is this paper novel?

## Key takeaways

- Canonical Correlation Analysis (CCA) - Two sets of learned weights that when applied to X and Y maximizes their correlation.
- Average Precision (AP) - Average probability of items that should be 1.
- Probabilistic Cross-Modal Embedding (PCME) - Similar to VAE. Predict mean and std then sample from it to get embedding.
- Soft crossmodal contrastive loss - Given two vectors, output the likelihood of match
- Montecarlo L2 loss - Using PCME sample many points and do L2 loss on vectors. Average the loss.

## What I still do not understand?

## Ideas to pursue

## Similar papers
