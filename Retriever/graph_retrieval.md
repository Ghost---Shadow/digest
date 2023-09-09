# [Maximum Common Subgraph Guided Graph Retrieval: Late and Early Interaction Networks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf7a83a5342befd11d3d65beba1be5b0-Abstract-Conference.html)

**Source Code:**

**Datasets:**

**Author:** Indian Institute of Technology Bombay

**Journal:** Neurips

**Year of Submission:** 2022

## What problem does it solve?

## How does it solve it?

### Training flow

### Inference

### Equations

```txt
P = GS(H_q.T @ H_c)
```

- P is permutation matrix
- H_q embeddings for each node in the query graph, stacked. shape = [num_nodes, node_embedding_dim]
- H_c embeddings for each node in each of the graphs in the corpus, stacked. shape = [corpus_size, num_nodes, node_embedding_dim]

```txt
S(G_q, H_q) = min(H_q, P @ H_c)
```

- S is scoring function
- The authors approximate H as the adjacency matrix. It is stacked node embeddings.

### Model

## How is this paper novel?

## List of experiments

### Ablation Studies

### Efficiency analysis

## Preliminaries

### MCES (subgraph may not be connected)

```py
max_common_edges = 0
for P in all_permutations:
    common_edge_count = 0
    # Ac is adjacency matrix of corpus graph
    permuted_Ac = P @ Ac @ P.T  # Matrix multiplication
    for i in range(N):  # N is the size of the adjacency matrices
        for j in range(N):
            common_edge_count += min(Aq[i][j], permuted_Ac[i][j])
    max_common_edges = max(max_common_edges, common_edge_count)
```

1. `Aq` and `Ac`: These are the adjacency matrices of two graphs, say `Gq` (query graph) and `Gc` (corpus graph). An adjacency matrix represents which nodes in a graph are adjacent to each other.
2. `P`: This represents a permutation matrix. Multiplying an adjacency matrix `Ac` by `P` from the left and `P^T` from the right permutes its rows and columns, essentially reordering the nodes of the graph.
3. `min(Aq[i,j], (P*Ac*P^T)[i,j])`: For each element `(i,j)` in the matrix, this takes the minimum of the corresponding elements in `Aq` and the permuted matrix `P*Ac*P^T`. This operation effectively identifies the edges that are common between the two graphs under a given node permutation.
4. `sum(i,j)`: This summation is over all the elements of the resulting matrix from the previous step, effectively counting the number of common edges between the two graphs under the given permutation.
5. `max P in P`: This maximization searches over all possible permutation matrices `P` to find the one that gives the largest number of common edges between `Gq` and `Gc`.

### Strongly connected graph

For all nodes, in a directed graph if there exists a path from one to another, then the graph is strongly connected graph.

### TarjanSCC

Given a directed graph finds its strongly connected components.

### MCCS (subgraph must be connected)

```python
def max_connected_subgraph(Aq, Ac):
    """Find the maximum connected common subgraph size 
    between adjacency matrices Aq and Ac."""
    
    N = Aq.shape[0]
    max_size = 0

    for P in all_possible_permutations(N):
        common_subgraph_matrix = matrix_min(Aq, P @ Ac @ P.T)
        size = tarjans_scc(common_subgraph_matrix)
        if size > max_size:
            max_size = size

    return max_size
```

### Gumbel sinkhorn network

Gumbel softmax

```python
ret = y_hard - y_soft.detach() + y_soft
```

Sinkhorn iteration

```python
def sinkhorn_iteration(matrix, max_iterations=100, tolerance=1e-3):
    """
    Normalize a matrix using the Sinkhorn operation to make it approximately doubly stochastic.
    
    Args:
    - matrix (np.array): The input matrix to normalize.
    - max_iterations (int): Maximum number of iterations for the Sinkhorn normalization.
    - tolerance (float): Convergence tolerance. The operation will stop if changes between 
      consecutive matrices are below this threshold.

    Returns:
    - np.array: The normalized matrix.
    """
    for _ in range(max_iterations):
        # Row normalization - make rows sum up to 1
        matrix /= matrix.sum(axis=1, keepdims=True)
        
        # Column normalization - make columns sum up to 1
        matrix /= matrix.sum(axis=0, keepdims=True)
        
        # Check for convergence
        row_sums = matrix.sum(axis=1)
        col_sums = matrix.sum(axis=0)

        if np.all(np.abs(row_sums - 1) < tolerance) and np.all(np.abs(col_sums - 1) < tolerance):
            break

    return matrix
```

## GPU hours

## Key takeaways

## What I still do not understand?

- what is the error margin?
- node embeddings is adjacency matrix?

## Ideas to pursue

## Similar papers
