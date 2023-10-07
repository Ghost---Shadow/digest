# [Active Learning Through a Covering Lens](https://arxiv.org/abs/2205.11320)

**Source Code:** https://github.com/avihu111/TypiClust

**Datasets:** CIFAR10/ImageNet

**Author:** The Hebrew University of Jerusalem

**Journal:** Neurips

**Year of Submission:** 2022

**Youtube:**

## What problem does it solve?

## How does it solve it?

1. Semantically embed all points
2. Pick the points that have most neighbours within a sphere

### Training flow

### Inference

### Equations

#### Assumptions

* Purity Assumption: KNN of a point within a sphere should have the same label.

#### Expectation of wrong labeling with purity assumption

Expectation of wrong labeling with purity assumption `<=` The probability that a given label is unreachable or outside the covered region + The probability that the label is misclassified within the covered region due to impurity

#### argmax (for L subset of X and size of L equals b) P( union (for each x in L) Bδ(x) )

* "argmax" means you're trying to find the set L that maximizes the following function.
* "L subset of X" denotes that L is a subset from the unlabeled set of points X.
* "size of L equals b" is a constraint ensuring that the subset L contains b points.
* "P(...)" represents the probability function.
* "union" symbolizes the combined or unified area or set.
* "Bδ(x)" represents a ball (or circle) of radius δ centered at x.

#### Max Probability Cover problem

`P(union from i=1 to b of Bδ(xi)) = 1/|X| * |{x in X | there exists i such that distance between xi and x is less than δ}| = 1/|X| * sum from i=1 to b of |Bδ(xi) intersect X|`

`P(union from i=1 to b of Bδ(xi))`
This represents the probability of the combined region of `b` balls of radius `δ` centered at each `xi`.

`1/|X| * |{x in X | there exists i such that distance between xi and x is less than δ}|`
Here, you're determining the size of the set of all points `x` from `X` that lie within distance `δ` from any of the labeled points `xi`. By dividing it by `|X|`, you're calculating the fraction of the total points that are within any of these `δ`-balls.

`1/|X| * sum from i=1 to b of |Bδ(xi) intersect X|`
For each `xi`, you're determining the size of the intersection between the `δ`-ball centered at `xi` and the set `X`. This provides the count of points from `X` that are within distance `δ` of that specific `xi`. Summing up across all `xi` gives the total number of points in `X` that fall within any of the `δ`-balls. Dividing by `|X|` converts this count to a fraction of the total points.

In simpler terms, the aim is to select points `xi` that, within their `δ`-radius balls, cover the maximum number of neighbors or points from `X`.

```python
import networkx as nx

def prob_cover(unlabeled_pool, labeled_pool, budget, ball_size, embedding_function):
    """
    :param unlabeled_pool: List of unlabeled data
    :param labeled_pool: List of labeled data
    :param budget: Number of samples to be selected
    :param ball_size: Size of the δ-ball
    :param embedding_function: Function to embed data into desired space
    :return: List of samples to query
    """
    
    # Step 1: Embed the data
    all_data = unlabeled_pool + labeled_pool
    embeddings = embedding_function(all_data)

    # Step 2: Construct the graph
    graph = nx.DiGraph()

    for i, embedding in enumerate(embeddings):
        graph.add_node(i, embedding=embedding)

    for i, source_embedding in enumerate(embeddings):
        for j, target_embedding in enumerate(embeddings):
            distance = compute_distance(source_embedding, target_embedding)
            
            if distance <= ball_size and i != j:
                graph.add_edge(i, j)

    # Step 3: Remove edges for labeled samples
    for labeled_sample in labeled_pool:
        index = all_data.index(labeled_sample)
        graph.remove_edges_from(list(graph.in_edges(index)))

    # Step 4: Greedy selection
    queries = []
    for _ in range(budget):
        max_out_degree_node = max(graph.nodes(), key=lambda node: graph.out_degree(node))
        queries.append(all_data[max_out_degree_node])
        
        # Remove edges
        neighbors = list(graph.neighbors(max_out_degree_node))
        for neighbor in neighbors:
            graph.remove_edge(max_out_degree_node, neighbor)

    return queries

def compute_distance(embedding1, embedding2):
    """
    Compute the distance between two embeddings. 
    This can be adapted depending on the desired distance metric.
    """
    return sum((e1 - e2)**2 for e1, e2 in zip(embedding1, embedding2))**0.5
```

### Model

## How is this paper novel?

## List of experiments

### Ablation Studies

### Efficiency analysis

## Preliminaries

## GPU hours

## Key takeaways

## What I still do not understand?

## Ideas to pursue

## Similar papers
