# [TODO] [Diverse Demonstrations Improve In-context Compositional Generalization](https://arxiv.org/pdf/2212.06800.pdf)

## Meta

* Journal -
* Year - 2023
* Author - Tel-Aviv University
* Code - https://github.com/itayle/diverse-demonstrations
* One liner -
* Model - T5 base, T5 11B, T5 large, Palm 62B, Palm 540B
* Datasets - SMCalFlow, GeoQuery, COVR-10
* Baselines - Random, Top-K, DPP

## Algorithm

```python
def cover_ls_algorithm(s, t, retriever, k):
    """
    Cover-LS Algorithm

    :param s: List of candidate local structures to cover
    :param t: Pool of training examples
    :param retriever: Function that takes a structure and the training pool and returns an example containing the structure
    :param k: Desired number of output examples
    :return: Set of training examples d
    """
    d = set()  # Initialize d as an empty set
    s.sort(reverse=True)  # Sort s from largest to smallest
    
    while len(d) < k:
        s_uncovered = s.copy()  # Start with all structures as uncovered
        for structure in s_uncovered:
            example = retriever(structure, t)  # Retrieve an example that contains the structure
            if example:
                d.add(example)  # Add example to d
                # Assuming `remove_covered_structures` and `remove_examples_with_same_anon_program` are implemented
                s_uncovered = remove_covered_structures(s_uncovered, example)  # Update s_uncovered
                t = remove_examples_with_same_anon_program(t, example)  # Remove examples with same anon program
            if len(d) == k:
                break  # Exit early if d has reached desired size
    
    return d
```

## Experiments

* Method vs LLM accuracy
* Number of shots vs LLM accuracy for no finetuning setup
* Number of shots vs LLM accuracy for finetuning with T5 (?)
* Number of beam search candidates
* STD for FT setup for 3 seeds
* STD for NoFT setup for 3 seeds
