# IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Large Language Models

## Meta

- **Name**: IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Large Language Models
- **Journal**: ICLR (International Conference on Learning Representations)
- **Year**: 2024
- **Authors**: Pennsylvania State University, The University of Sydney, Tsinghua University, Xidian University
- **Code**: [IDEAL on GitHub](https://skzhang1.github.io/IDEAL/)
- **One-liner**: An influence-driven selective annotation method to reduce annotation costs while improving the selection of prompts for large language models.
- **Model**: GPT-J 6B, GPT-Neo 2.7B, Text-davinci-002, GPT-3.5-Turbo
- **Datasets**: MRPC, SST-5, MNLI, DBpedia, RTE, HellaSwag, MWoZ, GeoQuery, Xsum
- **Baselines**: Random selection, Vote-k, K-Means, Maximizing facility location (MFL), Fast Vote-k

## Formulas

This section provides a breakdown of the key formulas and variables used in the paper, expressed using MathJax-style LaTeX.

### 1. Graph Representation and Influence Function

The paper considers a graph:

$$
G = (V, E, P)
$$

- $ V $: The set of vertices representing candidates or retrieval set elements.
- $ E $: The set of edges representing relationships or similarities.
- $ P $: Probabilities associated with edges reflecting cosine similarity between embeddings.

The influence of a subset $ S \subseteq V $ is measured by $ f_G(S) $, quantifying the number of vertices activated when diffusion starts from $ S $ and spreads over $ G $.

### 2. Subset Influence Function and Influence Improvement

The influence improvement by adding a vertex $ v $ to a subset $ S $:

$$
\psi_v(S) = f_G(S \cup \{v\}) - f_G(S).
$$

- $ \psi_v(S) $: Incremental gain in influence when $ v $ is added to $ S $.
- Measures additional activation on the graph by adding $ v $.

### 3. Submodular Condition

The influence function $ f_G(S) $ satisfies submodularity:

$$
f_G(S_a \cup \{v\}) - f_G(S_a) \geq f_G(S_b \cup \{v\}) - f_G(S_b),
$$

for $ S_a \subset S_b $ and $ v \in V \setminus S_b $.

### 4. Marginal Gain Greedy Selection

To select a subset that maximizes influence:

$$
\max_{v \in V \setminus S_t} f_G(S_t \cup \{v\}),
$$

- At each iteration $ t $, choose $ v_t $ providing maximum marginal gain.

### 5. Approximation and Lower Bound Guarantees

- **Proposition 1 (Upper Bound)**:

  $$
  f_G(S^*_m) \leq f_G(S_t) + m\, \psi_{t+1}
  $$

- **Theorem 1 (Lower Bound)**:

  $$
  f_G(S_m) \geq \left(1 - \left(1 - \frac{1}{m}\right)^m\right) f_G(S^*_m)
  $$

### Summary

- Defines a graph-based structure with vertices (data points) and probabilistic edges (cosine similarity).
- Uses an influence function to quantify activation starting from a subset $ S $.
- Incorporates submodularity for greedy subset selection.
- Provides approximation bounds ensuring near-optimal performance.

## Training Flow

### Key Steps

1. **Directed Graph Construction**: Embed each instance using Sentence-BERT and construct a graph based on cosine similarity.

2. **Influence Quantification**: Simulate diffusion in graph $ G $ starting from a candidate subset $ S $.

3. **Subset Search with Maximum Influence**: Use a greedy algorithm to find a subset with maximum influence within budget $ m $.

4. **Annotation and Prompt Retrieval**: Manual annotation of $ S_u $ and retrieve prompts based on similarity.

5. **Theoretical Validation**: Validate lower bound for influence approximation.

6. **Efficiency Considerations**: Optimize graph density and minimize processing complexity.

### Pseudocode

```python
import torch
from sentence_transformers import SentenceTransformer
import random

model = SentenceTransformer('all-mpnet-base-v2')

def construct_graph(data, k=10):
    embeddings = model.encode(data, convert_to_tensor=True)
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    edges = []
    for i in range(len(data)):
        knn_indices = torch.topk(cosine_sim[i], k, largest=True).indices
        for j in knn_indices:
            if i != j:
                edges.append((i, j, cosine_sim[i][j]))
    return edges

def influence_quantification(graph, subset):
    influenced = set(subset)
    to_explore = list(subset)
    while to_explore:
        current = to_explore.pop()
        for neighbor in graph[current]:
            if neighbor not in influenced and random.random() < graph.edge_weights[(current, neighbor)]:
                influenced.add(neighbor)
                to_explore.append(neighbor)
    return len(influenced)

def greedy_search(graph, budget):
    selected = set()
    while len(selected) < budget:
        best_candidate, best_influence = None, 0
        for node in set(range(len(graph.x))) - selected:
            candidate_influence = influence_quantification(graph, selected | {node})
            if candidate_influence > best_influence:
                best_candidate, best_influence = node, candidate_influence
        selected.add(best_candidate)
    return selected
```

## Inference Flow

### Key Steps

1. Construct graph with vertices as embeddings and edges based on cosine similarity.
2. Quantify influence through candidate subset diffusion.
3. Use a greedy algorithm for maximum marginal influence gain.
4. Retrieve similar annotations for test inputs based on embeddings.

### Pseudocode

```python
import torch
import torch.nn.functional as F

class IDEALInference:
    def __init__(self, embeddings, k, annotation_budget):
        self.embeddings = embeddings
        self.k = k
        self.annotation_budget = annotation_budget

    def construct_graph(self):
        cos_sim = F.cosine_similarity(self.embeddings.unsqueeze(1), self.embeddings.unsqueeze(0), dim=2)
        _, indices = torch.topk(cos_sim, self.k+1, dim=-1)
        indices = indices[:, 1:]
        edge_weights = cos_sim.gather(1, indices)
        edge_weights /= edge_weights.sum(dim=1, keepdim=True)
        return indices, edge_weights

    def quantify_influence(self, candidate_subset):
        influence_scores = []
        for _ in range(10):
            influence_scores.append(self.diffusion_process(candidate_subset))
        return sum(influence_scores) / len(influence_scores)

    def diffusion_process(self, active_set):
        activated = active_set.clone()
        newly_activated = active_set.clone()
        while newly_activated.numel() > 0:
            new_active = torch.zeros_like(activated)
            for node in newly_activated:
                for neighbor, p in zip(indices[node], edge_weights[node]):
                    if not activated[neighbor] and torch.rand(1).item() < p:
                        new_active[neighbor] = 1
            newly_activated = new_active
            activated |= newly_activated
        return activated.sum().item()

    def select_maximum_influence_subset(self):
        selected_subset = []
        current_influence = 0
        while len(selected_subset) < self.annotation_budget:
            best_candidate = None
            best_gain = -1
            for i in range(len(self.embeddings)):
                if i in selected_subset:
                    continue
                candidate_influence = self.quantify_influence(torch.tensor(selected_subset + [i]))
                gain = candidate_influence - current_influence
                if gain > best_gain:
                    best_candidate = i
                    best_gain = gain
            selected_subset.append(best_candidate)
            current_influence += best_gain
        return selected_subset

    def retrieve_prompts(self, test_input):
        cos_sim = F.cosine_similarity(self.embeddings, test_input.unsqueeze(0), dim=1)
        _, top_indices = torch.topk(cos_sim, k=5)
        return top_indices
```

## Experiments

### List of Experiments

- Annotation budget ablations (Table 1)
- Time cost comparison between IDEAL and Vote-k (Figure 3)
- Candidate subset influence vs performance on in-context learning correlation (Figure 4)
- Visualization using UMAP (Figure 6)
- Performance with varying prompt orders (Table 10)
- Comparison with other coreset selection methods (Table 2)
- Evaluation with different retrieval methods (Table 3)
- Evaluation with different language models (Figure 5)
- Evaluation on out-of-distribution tasks (Table 4)
- Automatic annotation scenario (Table 5)
- Min max mean results across seeds (Tables 7 and 8)

## Proofs

### List of Proofs

1. **Submodularity of Influence Function**
   - The influence function $ f_G $ satisfies the submodular condition, ensuring diminishing returns when adding data points to smaller subsets.

2. **Lower Bound Termination Guarantee**
   - The subset $ S_m $ selected by the greedy algorithm achieves at least a fraction of the optimal solution's influence.

3. **Time Complexity Analysis**
   - The selection process is bounded by $ O(m \cdot (a + b)) $ where $ a $ is nodes and $ b $ is edges in graph $ G $. 

Refer to the full mathematical derivations in the paper for detailed proofs.