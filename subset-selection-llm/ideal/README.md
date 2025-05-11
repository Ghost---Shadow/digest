# IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Large Language Models

## Meta

- **Journal:** ICLR (International Conference on Learning Representations)
- **Year:** 2024
- **Authors:** Pennsylvania State University, The University of Sydney, Tsinghua University, Xidian University
- **Code:** [IDEAL GitHub](https://skzhang1.github.io/IDEAL/)
- **One-liner:** An influence-driven selective annotation method to decrease annotation costs while enhancing prompt selection for large language models.
- **Models:** GPT-J 6B, GPT-Neo 2.7B, Text-davinci-002, GPT-3.5-Turbo
- **Datasets:** MRPC, SST-5, MNLI, DBpedia, RTE, HellaSwag, MWoZ, GeoQuery, Xsum
- **Baselines:** Random selection, Vote-k, K-Means, Maximizing facility location (MFL), Fast Vote-k

## Formulas

### 1. Graph Representation and Influence Function

The paper introduces a graph structure represented as:

$$
G = (V, E, P)
$$

Where:
- $V$: The set of vertices, each representing a candidate data example or a retrieval set element.
- $E$: The set of edges denoting relationships or similarities between vertices.
- $P$: Probabilities associated with edges, reflective of \emph{cosine similarity} between embeddings, influencing the likelihood of information being passed during diffusion.

The influence of a subset $S \subseteq V$ is captured by $f_G(S)$, which measures the number of vertices activated when diffusion originates from $S$ across graph $G$.

### 2. Subset Influence Function and Influence Improvement

Influence improvement by adding a vertex $v$ to subset $S$ is defined as:

$$
\psi_v(S) = f_G(S \cup \{v\}) - f_G(S)
$$

Where:
- $\psi_v(S)$: Incremental influence gain from adding $v$ to $S$.
- $f_G(S \cup \{v\})$: Influence after including vertex $v$.
- $f_G(S)$: Influence before adding $v$.

### 3. Submodular Condition

The influence function $f_G(S)$ exhibits submodularity:

$$
f_G(S_a \cup \{v\}) - f_G(S_a) \geq f_G(S_b \cup \{v\}) - f_G(S_b)
$$

Where:
- $S_a \subset S_b$.
- $v \in V \setminus S_b$.

This inequality demonstrates the \emph{diminishing returns property} that justifies greedy subset selection strategies.

### 4. Marginal Gain Greedy Selection

To optimize influence selection under budget $m$:

$$
\max_{v \in V \setminus S_t} f_G(S_t \cup \{v\})
$$ 

Where:
- $S_t$: Current subset after $t$ iterations.
- $V \setminus S_t$: Remaining candidates.

The vertex $v_t$ maximizing the marginal gain is selected at each step.

### 5. Approximation and Lower Bound Guarantees

#### a. Proposition 1 (Upper Bound):

$$
f_G(S^*_m) \leq f_G(S_t) + m\, \psi_{t+1}
$$

Where:
- $S^*_m$: Optimal $m$-vertex subset.
- $\psi_{t+1}$: Maximum upcoming marginal improvement.

#### b. Theorem 1 (Lower Bound):

$$
f_G(S_m) \geq \left(1 - \left(1 - \frac{1}{m}\right)^m\right) f_G(S^*_m)
$$

Ensuring the greedy algorithm's influence is at least $(1 - (1 - \frac{1}{m})^m)$ times the optimal influence, converging to $1 - e^{-1}$ as $m$ grows.

## Training Flow

### Steps

1. **Directed Graph Construction:** Embed data using Sentence-BERT to create a directed graph $G$, connecting embeddings by cosine similarity.
2. **Influence Quantification:** Simulate diffusion to measure influence by counting activated vertices from subset $S$.
3. **Subset Search with Maximum Influence:** Utilize a greedy algorithm to select a maximally influential subset $S_u$ within budget $m$.
4. **Annotation and Prompt Retrieval:** Annotate $S_u$ and use embeddings for similarity-based prompt retrieval.
5. **Theoretical Validation:** Establish a lower bound for subset influence by the method.
6. **Efficiency Considerations:** Balance graph construction parameters and optimize for computational efficiency.

### Training Flow Code

```python
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
import random
import numpy as np

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
    return Data(edges=torch.tensor(edges), x=embeddings)

def influence_quantification(graph, subset):
    influenced = set(subset)
    to_explore = list(subset)
    while to_explore:
        current = to_explore.pop()
        for neighbor in graph.edges[current]:
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

data = ["sample text1", "sample text2", "..."]
graph = construct_graph(data)
selected_subset = greedy_search(graph, budget=10)
```

## Inference Flow

### Steps

1. **Graph Construction:** Create a directed graph from embeddings, connecting vertices by cosine similarity.
2. **Influence Quantification:** Initiate and simulate a diffusion process to measure subset influence.
3. **Greedy Selection for Maximum Influence:** Incrementally select vertices that maximize influence within the budget.
4. **Prompt Retrieval:** Fetch similar annotated entries for test inputs using cosine similarity.

### Inference Flow Code

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
- Time cost comparison between IDEAL and Vote-k during subset selection (Figure 3)
- Correlation between candidate subset influence vs. performance on in-context learning (Figure 4)
- Visualization of selected examples using UMAP (Figure 6)
- Performance under varying prompt order (Table 10)
- Comparison with other coreset selection methods (Table 2)
- Evaluation with different retrieval methods (Table 3)
- Evaluation with different language models (Figure 5)
- Evaluation on out-of-distribution tasks (Table 4)
- Automatic annotation scenario (Table 5)
- Min, max, mean results across seeds (Table 7 and Table 8)

## Proofs

### List of Proofs

1. **Submodularity of Influence Function**
   - Demonstrates that influence function $f_G$ meets the submodular condition, ensuring maximum influence gain by augmenting smaller subsets before larger ones.

2. **Lower Bound Termination Guarantee**
   - Shows that the greedy-selected subset $S_m$ reaches at least $(1 - (1 - 1/m)^m)$ of optimal influence $f_G(S^*_m)$, with performance nearing $1 - 1/e$ as annotation budget $m$ increases.

3. **Time Complexity Analysis**
   - Analyzes the overall time complexity for data selection, detailing diffusion processes and greedy selection scope based on graph dimensions $a$ nodes and $b$ edges, amounting to $O(m \cdot (a + b))$.

For detailed mathematical proofs, refer to relevant sections in the paper.