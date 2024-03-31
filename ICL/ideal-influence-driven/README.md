# [IDEAL: INFLUENCE-DRIVEN SELECTIVE ANNOTATIONS EMPOWER IN-CONTEXT LEARNERS IN LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2310.10873.pdf)

## Meta

* Journal - ICLR
* Year - 2024
* Author - Pennsylvania State University, The University of Sydney, Tsinghua University, Xidian University
* Code - https://github.com/xlang-ai/icl-selective-annotation
* One liner -
* Model - all-mpnet-base-v2, GPT-J 6B, GPT-Neo 2.7B
* Datasets - SST5, MRPC, MNLI, DBpedia, RTE, HellaSwag, MWoZ, Xsum
* Baselines - Random, [Vote-k and fast vote-k](https://arxiv.org/abs/2209.01975), K means, [maximum facility location](https://apps.dtic.mil/sti/citations/tr/ADA518795), 

## Training flow

## Algorithm

```python
import random

def subset_influence_quantification(graph, initial_subset):
    """
    Algorithm: Subset influence quantification.
    
    :param graph: Directed graph represented as a dict where keys are node identifiers
              and values are dicts of successors with edge probabilities.
              For example, graph[v] = {u: p} means there is an edge from v to u with probability p.
    :param initial_subset: Initial subset of influenced vertices.
    :return: Number of influenced vertices by initial_subset in graph.
    """
    active_subset = set(initial_subset)  # Active set of nodes to process
    new_subset = set()  # Newly influenced nodes
    influenced_count = 0  # Counter for influenced nodes
    
    while active_subset:
        for current_node in active_subset:
            for successor, probability in graph.get(current_node, {}).items():
                if successor not in initial_subset:
                    random_number = random.random()  # Generate a random number in [0, 1]
                    if random_number <= probability:
                        initial_subset.add(successor)
                        new_subset.add(successor)
        influenced_count += len(new_subset)
        active_subset, new_subset = new_subset, set()  # Prepare for the next iteration
    
    return influenced_count

# Example usage
graph = {
    'a': {'b': 0.5, 'c': 0.2},
    'b': {'c': 0.7},
    'c': {'d': 0.9},
    'd': {}
}
initial_subset = {'a'}

influenced_vertices_count = subset_influence_quantification(graph, initial_subset)
print("Number of influenced vertices:", influenced_vertices_count)
```

## Inference flow

1. Compute all to all similarity matrix
2. Start with a seed set
3. For each item in seed set pick the top K most similar items
4. Add these back into the seed set with probability inversely proportional to their similarity with the seed set (uniform random) (semantic deduplication)
5. Continue until desired size

## List of experiments

* Annotation budget ablations
* Time cost comparison
* candidate score vs performance on downstream task correlation
* Visualization of clusters
* Min max mean results accross seeds

## List of proofs

* Submodularity of influence function
* Lower bound termination guarantee
* Time complexity analysis
