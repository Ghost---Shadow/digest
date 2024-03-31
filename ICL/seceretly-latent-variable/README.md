# [TODO] [Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning](https://arxiv.org/pdf/2301.11916.pdf)

## Meta

* Journal - NeurIPS
* Year - 2023
* Author - University of California
* Code - https://github.com/WANGXinyiLinda/concept-based-demonstration-selection
* One liner - Learning a softprompt using ICL (?)
* Model -
* Datasets -
* Baselines -

## Algorithm

```python
import torch

def latent_concept_learning(dataset, tasks, model, concept_tokens_per_task, learning_rate, training_steps):
    """
    Latent Concept Learning Algorithm

    Parameters:
    - dataset: Dataset as a list of tuples (x_i, y_i, d_i)
    - tasks: Set of tasks
    - model: Pre-trained Language Model (LLM)
    - concept_tokens_per_task: Number of concept tokens per task
    - learning_rate: Learning rate
    - training_steps: Number of training steps
    """
    # Add concept_tokens_per_task * |tasks| new tokens to the vocabulary and initialize their embeddings
    e_new = torch.randn(concept_tokens_per_task * len(tasks), requires_grad=True)  # Random initialization
    model.freeze_parameters()  # Freeze all parameters in model except e_new
    
    # Training loop
    for step in range(training_steps):
        gradient_accumulator = 0  # Initialize gradient accumulation variable
        batch = sample_random_batch(dataset)  # Assuming a function to sample a batch from dataset
        
        for x, y, d in batch:
            # Forward pass to compute loss, assuming a loss function and a method to integrate e_new
            loss = compute_loss(model, x, y, d, e_new)
            
            # Accumulate gradients
            gradient_accumulator += torch.autograd.grad(loss, e_new, retain_graph=True)[0]
        
        # Update e_new using the accumulated gradients
        with torch.no_grad():
            e_new -= learning_rate * gradient_accumulator

    # Assuming some method to integrate e_new back into model for the final model M'
    model_prime = integrate_new_embeddings(model, e_new)
    return model_prime

```

```python
import torch

def select_demonstrations(dataset_d, model_prime, k):
    """
    Demonstration Selection Algorithm

    Parameters:
    - dataset_d: Dataset D_d for a task d, as a list of tuples (X_d, Y_d)
    - model_prime: LLM with fine-tuned concept tokens M'
    - k: The number of demonstrations to select

    Output:
    - A set of selected demonstrations
    """
    scores = []
    demonstrations = []

    # Compute score for each (X_d, Y_d) in dataset_d
    for X_d, Y_d in dataset_d:
        score = compute_probability(model_prime, X_d, Y_d)
        scores.append(score)
        demonstrations.append((X_d, Y_d))

    # Select top k examples with the largest scores
    selected_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    selected_demonstrations = [demonstrations[i] for i in selected_indices]

    return selected_demonstrations
```

## Proofs

* ICL can be bayes optimal predictor (?) with finite shots (?)
* Bayes optimal classifier is better than in context classifier (?)

## Experiments

* Accuracy vs model vs random shot or similar shot
* Anti-causal (?) direction
* Random words and random labels using GPT2-large (?)
