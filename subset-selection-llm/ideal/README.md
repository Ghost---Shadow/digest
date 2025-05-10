# IDEAL: INFLUENCE-DRIVEN SELECTIVE ANNOTATIONS EMPOWER IN-CONTEXT LEARNERS IN LARGE LANGUAGE MODELS

## Meta

* **Name:** IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Large Language Models
* **Journal:** ICLR (International Conference on Learning Representations)
* **Year:** 2024
* **Author Institutions:** Pennsylvania State University, The University of Sydney, Tsinghua University, Xidian University
* **Code:** [GitHub Repository](https://skzhang1.github.io/IDEAL/)
* **One-liner:** An influence-driven selective annotation method to reduce annotation costs while enhancing prompt selection for large language models.
* **Models:** GPT-J 6B, GPT-Neo 2.7B, Text-davinci-002, GPT-3.5-Turbo
* **Datasets:** MRPC, SST-5, MNLI, DBpedia, RTE, HellaSwag, MWoZ, GeoQuery, Xsum
* **Baselines:** Random selection, Vote-k, K-Means, Maximizing facility location (MFL), Fast Vote-k

## Formulas

Below is a detailed breakdown of each variable and term used in the provided formulas, with explanations written in MathJax-style LaTeX.

1. **Edge Probability**  
   \[
   p(v,u) = \frac{\cos(v,u)}{\sum_{z \in N(v,k)} \cos(v,z)}
   \]
   - \(\displaystyle p(v,u)\): The probability (or weight) assigned to the directed edge from vertex \(v\) to vertex \(u\).
   - \(\displaystyle \cos(v,u)\): The cosine similarity between the embeddings (feature representations) of vertices \(v\) and \(u\).
   - \(\displaystyle N(v,k)\): The set of \(k\) nearest neighbors of vertex \(v\).
   - \(\displaystyle \sum_{z \in N(v,k)} \cos(v,z)\): Normalizes cosine similarities in \(N(v,k)\).

2. **Optimal Subset Selection**  
   \[
   S^*_m = \arg\max_{S \subset V} f_G(S), \quad \text{s.t.} \; |S| = m
   \]
   - \(\displaystyle S^*_m\): Optimal subset of vertices of size \(m\).
   - \(\displaystyle f_G(S)\): Total influence initiated by subset \(S\).

3. **Submodularity Condition**  
   \[
   f_G(S_a \cup \{v\}) - f_G(S_a) \geq f_G(S_b \cup \{v\}) - f_G(S_b)
   \]
   - Demonstrates diminishing returns in influence as subsets grow large.

4. **Greedy Approximation Bound**  
   \[
   f_G(S_m) \geq \left(1 - \left(1 - \frac{1}{m}\right)^m\right)f_G(S^*_m)
   \]
   - Approaches \(1 - \frac{1}{e}\) (approximately 0.632) for large \(m\).

5. **Subset Influence**  
   \[
   L = \left| \{ \text{Influenced vertices by } S \} \right|
   \]
   - \(\displaystyle L\): Number of vertices influenced by subset \(S\).

6. **Algorithmic Step: Maximum Influence Search**  
   - Iteratively selects vertex \(v\) maximizing the marginal influence gain:
     \[
     \Delta f_G(v \mid S) = f_G(S \cup \{v\}) - f_G(S)
     \]

## Training Flow

1. **Directed Graph Construction**: Compute embeddings for each instance. Create a graph based on cosine similarity, assigning weights proportionally.
2. **Influence Quantification**: Simulate diffusion processes to quantify a subset's influence.
3. **Subset Search**: Apply greedy algorithm to find the subset with maximum influence.
4. **Annotation & Prompt Retrieval**: Annotate the selected subset; use cosine similarity for prompt retrieval.
5. **Theoretical Validation**: Establish lower bounds for influence approximations.
6. **Efficiency Considerations**: Optimize computational processes for scalability.

### Pseudocode
```python
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
import random
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')

def construct_graph(data, k=10):
    embeddings = model.encode(data, convert_to_tensor=True)
    cosine_sim = torch.nn.functional.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
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
            candidate_influence = influence_quantification(
                graph, selected | {node})
            if candidate_influence > best_influence:
                best_candidate, best_influence = node, candidate_influence
        selected.add(best_candidate)
    return selected

data = ["sample text1", "sample text2", "..."]
graph = construct_graph(data)
selected_subset = greedy_search(graph, budget=10)
```

## Inference Flow

1. **Graph Construction**: Create graph with vertices as embeddings and edges for \(k\) nearest neighbors.
2. **Influence Quantification**: Measure subset influence through diffusion processes.
3. **Greedy Subset Selection**: Choose vertices providing maximum marginal influence gain.
4. **Prompt Retrieval**: Retrieve annotated examples most similar to test inputs.

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
        # Compute cosine similarities and determine the k nearest neighbors
        cos_sim = F.cosine_similarity(self.embeddings.unsqueeze(1), self.embeddings.unsqueeze(0), dim=2)
        _, indices = torch.topk(cos_sim, self.k+1, dim=-1)  # +1 to include self in neighbors
        indices = indices[:, 1:]  # Exclude self
        
        # Prepare edge weights based on cosine similarities
        edge_weights = cos_sim.gather(1, indices)
        edge_weights /= edge_weights.sum(dim=1, keepdim=True)
        return indices, edge_weights

    def quantify_influence(self, candidate_subset):
        # Run diffusion process multiple times and average results
        influence_scores = []
        for _ in range(10):  # Repeat process to stabilize result
            influence_scores.append(self.diffusion_process(candidate_subset))
        return sum(influence_scores) / len(influence_scores)

    def diffusion_process(self, active_set):
        # Independent cascade model simulation
        activated = active_set.clone()
        newly_activated = active_set.clone()
        while newly_activated.numel() > 0:
            # Iteratively activate neighbors
            new_active = torch.zeros_like(activated)
            for node in newly_activated:
                # Activate neighbors based on node connections and edge weights
                for neighbor, p in zip(indices[node], edge_weights[node]):
                    if not activated[neighbor] and torch.rand(1).item() < p:
                        new_active[neighbor] = 1
            newly_activated = new_active
            activated |= newly_activated
        return activated.sum().item()

    def select_maximum_influence_subset(self):
        # Greedy selection based on influence
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
        # Retrieve similar annotated examples based on cosine similarity
        cos_sim = F.cosine_similarity(self.embeddings, test_input.unsqueeze(0), dim=1)
        _, top_indices = torch.topk(cos_sim, k=5)  # Retrieve top 5 most similar
        return top_indices
```

## Experiments

### List of Experiments

* Annotation budget ablations (Table 1)
* Time cost comparison between IDEAL and Vote-k during subset selection (Figure 3)
* Candidate subset influence vs. performance on in-context learning correlation (Figure 4)
* Visualization of selected examples using UMAP (Figure 6)
* Performance under varying prompt order (Table 10)
* Comparison with other coreset selection methods (Table 2)
* Evaluation with different retrieval methods (Table 3)
* Evaluation with different language models (Figure 5)
* Evaluation on out-of-distribution tasks (Table 4)
* Automatic annotation scenario (Table 5)
* Min-max mean results across seeds (Table 7 and Table 8)

## Proofs

### List of Proofs

1. **Submodularity of Influence Function**
   - The influence function \( f_G \) satisfies the submodular condition: for any \( v \in V \), \( \forall S_a \subset S_b \subset V \), the submodularity condition \( f_G(S_a \cup v) - f_G(S_a) \geq f_G(S_b \cup v) - f_G(S_b) \) holds, indicating diminishing returns.

2. **Lower Bound Termination Guarantee**
   - The subset \( S_m \) selected by the greedy algorithm satisfies \( f_G(S_m) \geq (1 - (1 - 1/m)^m) f_G(S^*_m) \), approaching \( 1 - 1/e \).

3. **Time Complexity Analysis**
   - For graph \( G(a, b) \), with \( a \) nodes and \( b \) edges, and annotation budget \( m \), time complexity for diffusion is \( O(a+b) \) and overall algorithm is \( O(m \cdot (a + b)) \). 

For full mathematical derivations, please refer to the corresponding sections in the original paper.