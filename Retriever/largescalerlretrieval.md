# [Large-Scale Retrieval for Reinforcement Learning](https://arxiv.org/abs/2206.05314)

**Source Code:**

**Datasets:**

**Author:** Deepmind

**Journal:** Neurips

**Year of Submission:** 2022

## What problem does it solve?

Retrieval augmented go

## How does it solve it?

### Training flow

```python
for _ in range(num_gradient_steps):
    # y has action and value, o is board state
    o_t, y_t = sample_minibatch(D)

    # Embed the current board state 
    # (details are not mentioned in the paper)
    q_t = key_query_net(o_t)

    # x is an array of (board state, next 10 actions, game outcome, final board)
    x_t = get_neighbors(q_t, D_r)

    # a_tilde is valid actions from that board state (maybe)
    y_pred = model(o_t, a_tilde, x_t)
    
    loss = model.loss_fn(y_pred, y_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Inference

1. Embed the board state into a query vector
2. Lookup k-NN database
3. Database returns N neighbours. Retrieved neighborus include board, next 10 actions, game outcome, final board.
4. Neighbours are fed into an RNN which returns policy and value

### Equations

### Model

## How is this paper novel?

## List of experiments

## Preliminaries

## GPU hours

## Key takeaways

## What I still do not understand?

## Ideas to pursue

## Similar papers
