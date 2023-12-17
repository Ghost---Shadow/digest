# [A Neural Corpus Indexer for Document Retrieval](https://arxiv.org/abs/2206.02743)

**Source Code:** <https://github.com/athresh/orient>

**Datasets:** Office-31

**Author:** utdallas

**Journal:** NeurIPS

**Year of Submission:** 2022

## What problem does it solve?

Data subset selection in the source dataset which maximizes accuracy on an unseen target dataset.

It is solving the model capacity problem, because learning only things similar to target dataset frees up model capacity to specialize in that task.

## How does it solve it?

Pick datapoints from source dataset which give similar weight updates (gradient) in the target dataset.

Lets say source dataset is images of all animals and target is only birds. The images which would bear similar gradients as birds should be other pictures of birds. Since the model is now more specialized it should do better.

### Training flow

```python
def train_orient(source_data, target_data, model, loss_function, smi_function, total_epochs, subset_interval, batch_size):
    # Initialize a random subset A from source_data
    A_indices = random.sample(range(len(source_data)), batch_size)
    A = Subset(source_data, A_indices)

    # Define an optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(total_epochs):
        # Mini-batch gradient descent
        model.train()
        data_loader = DataLoader(A, batch_size=batch_size, shuffle=True)
        for batch in data_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        # Subset selection every subset_interval epochs
        if epoch % subset_interval == 0:
            A = select_subset(model, source_data, target_data, smi_function, batch_size)

    return model

def smi_function(model, source_data, target_data, batch_size):
    # Function to compute gradient norms for a batch of data
    def compute_gradients(data_loader):
        gradients = []
        for inputs, _ in data_loader:
            inputs = inputs.requires_grad_()
            outputs = model(inputs)
            model.zero_grad()
            outputs.backward(torch.ones_like(outputs))
            gradients.append(inputs.grad.norm(dim=1))
        return torch.cat(gradients)

    # Create data loaders
    source_loader = DataLoader(source_data, batch_size=batch_size)
    target_loader = DataLoader(target_data, batch_size=batch_size)

    # Compute gradients
    source_gradients = compute_gradients(source_loader)
    target_gradients = compute_gradients(target_loader)

    # Compute pairwise cosine similarity between source and target gradients
    similarity_matrix = torch.nn.functional.cosine_similarity(source_gradients.unsqueeze(0), target_gradients.unsqueeze(1), dim=2)

    # Compute SMI - this is a placeholder for the actual SMI computation
    # You would need to define how the SMI is calculated based on the similarity_matrix
    # For simplicity, let's just select the top indices based on average similarity
    average_similarity = similarity_matrix.mean(dim=1)
    selected_indices = average_similarity.topk(batch_size).indices

    return selected_indices.tolist()
```

### Equations

smi_function

### Model

ResNet50

## How is this paper novel?

The experiments showed that ORIENT could achieve over 2.5× speed-up in training time.

## List of experiments

### Ablation Studies

Sure, here's the revised list with bold headings for better readability:

- **Comparison of Different SMI Functions**: Evaluates various Submodular Mutual Information (SMI) functions within ORIENT, including ORIENT(GCMI), ORIENT(LDMI), ORIENT(FLMI), ORIENT(GM), and ORIENT(G), to assess their impact on target domain accuracy and performance-speedup trade-off.
- **Performance on Different Datasets**: Assesses the effectiveness of ORIENT on two domain adaptation datasets, Office-31 and Office-Home, to understand its performance under different dataset characteristics.
- **Effect of Subset Selection Interval (L)**: Investigates the impact of varying the subset selection interval \( L \) – the frequency at which the training subset \( A \) is updated.
- **Effect of Subset Sampling Rate (b)**: Examines the influence of changing the subset sampling rate \( b \) – the proportion of the source domain used for training – on performance.
- **Training Time Reduction Analysis**: Compares the training time of ORIENT with other methods like full training, random subset selection, and CRAIG, focusing on quantifying the speed-up achieved by ORIENT.
- **Performance with Different Loss Functions**: Assesses ORIENT's performance when combined with different SDA loss functions, such as d-SNE and CCSA.
- **Synthetic Experiments for SMI Function Data Subset Selection**: Provided in Appendix A.6, this study offers insights into how different SMI functions select data subsets, enhancing understanding of their practical application behavior.

## Preliminaries

## GPU hours

N/A

## Key takeaways

1. **Efficiency in Training**: ORIENT significantly reduces the training time compared to traditional training methods on the full source dataset. This efficiency is a crucial advantage, especially in resource-constrained environments.

2. **Performance Maintenance**: Despite the reduction in training time, ORIENT maintains comparable, and in some cases, superior performance to training on the complete source data. This balance between efficiency and effectiveness is a notable feature of the framework.

3. **Best Performing SMI Functions**: Among the different Submodular Mutual Information (SMI) functions tested, ORIENT(FLMI) consistently achieves the best performance versus speed-up trade-off. This finding is essential for selecting the most effective SMI function within the ORIENT framework.

4. **Compatibility with Existing SDA Methods**: ORIENT can augment existing domain adaptation methods, such as d-SNE and CCSA loss functions. This compatibility demonstrates the framework's versatility and potential for broader applicability in various SDA scenarios.

5. **Generalization of Previous Subset Selection Approaches**: ORIENT generalizes and extends previous subset selection methods like GLISTER, GRADMATCH, and CRAIG, showing that these methods can be viewed as instances of SMI-based subset selection.

6. **Dataset Adaptability**: The framework's effectiveness is validated on two domain adaptation datasets, Office-31 and Office-Home, indicating its adaptability to different types of data and domain shift scenarios.

7. **Methodological Innovations**: The use of last-layer gradient approximations for constructing the similarity matrix in the ORIENT framework illustrates an innovative approach to handling high-dimensional data in deep learning models.

8. **Substantial Speed-ups**: In some experimental settings, ORIENT achieves up to 3 times faster training than full dataset training methods, demonstrating substantial gains in computational efficiency.

9. **Detailed Ablation Studies**: The paper includes various ablation studies, such as the impact of different SMI functions, the effect of subset selection interval and subset sampling rate, which provide deeper insights into the framework's functionality and optimization.

10. **Open-Source Implementation**: The availability of ORIENT's implementation online encourages further exploration and adaptation in the research community, facilitating reproducibility and collaborative improvements.

## What I still do not understand?

## Ideas to pursue

## Similar papers
