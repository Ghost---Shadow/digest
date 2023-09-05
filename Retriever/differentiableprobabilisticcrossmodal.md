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

#### ASP

ASPq ​ =1/omega * sum(​Max(R(i,C),R(i,S)) / Min(R(i,C),R(i,S))​)

- q is query
- omega - retrieved set
- C semantic relevance between q and omega (CIDEr metric)
- S semantic similarity between q and omega (cosine similarity of semantic embeddings)
- R(i,C) - Ranking of an item i in list C

#### ASP differentiable

```python
def compute_DU(u):
    m = u.size(0)
    
    # Extend u into matrix form
    u_row = u.unsqueeze(0).repeat(m, 1)
    u_col = u.unsqueeze(1).repeat(1, m)
    
    # Compute D_U by subtracting the two matrices
    DU = u_row - u_col
    
    return DU


def ASPq(DC, DS):
    # ϕ is the sigmoid function
    ϕ = torch.nn.Sigmoid()
    
    # Calculate ϕ applied to each element of DC and DS
    phi_DC = ϕ(DC)
    phi_DS = ϕ(DS)
    
    # Sum along the rows for each matrix (assuming i is the row index and j is the column index)
    sum_phi_DC = torch.sum(phi_DC, dim=1)
    sum_phi_DS = torch.sum(phi_DS, dim=1)
    
    # Calculate the minimum and maximum between the summed values for each row
    min_values = torch.min(sum_phi_DC, sum_phi_DS)
    max_values = torch.max(sum_phi_DC, sum_phi_DS)
    
    # Compute the ASPq value for each row (i.e., each instance in Ω) and then average
    ASPq_values = 1 + min_values / (1 + max_values)
    average_ASPq = torch.mean(ASPq_values)
    
    return average_ASPq
```

- D is a matrix such that `D[i][j] = u[i] - u[j]`

### Model

## How is this paper novel?

## List of experiments

## Preliminaries

- Canonical Correlation Analysis (CCA) - Two sets of learned weights that when applied to X and Y maximizes their correlation.
- Average Precision (AP) - Average probability of items that should be 1.
- Probabilistic Cross-Modal Embedding (PCME) - Similar to VAE. Predict mean and std then sample from it to get embedding.
- Soft crossmodal contrastive loss - Given two vectors, output the likelihood of match
- Montecarlo L2 loss - Using PCME sample many points and do L2 loss on vectors. Average the loss.
- CIDEr - cosine similarity of TF-IDF feature vectors between query and each element in database

## Key takeaways

## What I still do not understand?

- what is PMRP?
- what is CxC?

## Ideas to pursue

## Similar papers
