# Reinforcement Learning

## [Learning a SAT Solver from Single-Bit Supervision](https://arxiv.org/abs/1802.03685)

**Source Code:** [Github](https://github.com/dselsam/neurosat)

**Datasets:** Generated (Same repo)

**Author:** (Stanford) Daniel Selsam, Matthew Lamm, Benedikt Bünz, Percy Liang, Leonardo de Moura, David L. Dill

**Year of Submission:** 2018

**Time to read (minute):** 100+ (Dropped)

**Easy to read?** No. Math notations too difficult.

### What problem does it solve?

A SAT solver which is trained on simpler problems can solve more complex problems

### How does it solve it?

- Message passing neural network MPNN.
- Single bit supervision to check satisfiability
- Solution can be decoded from the NN's activations

SAT is encoded in CNF.

#### Dataset

#### Model

- Each literal and clause is a vector.
- Each update for a clause, passes messages to neighbouring literal and then vice versa

```
Parameters of model

L[init]
C[init]
L[init] and C[init] are tiled to form L_t and C_t
Flip = swap each row of L_t corresponding to the literal's negation

Three MLPs
L[message]
C[message]
L[vote]

one LSTM with hidden states
L[u]
C[u]
```
Iteration
```
(C(t+1);C(t+1)h )   Cu([C(t)h ;M>Lmsg(L(t))])
(L(t+1);L(t+1)h )   Lu([L(t)h ; Flip(L(t));MCmsg(C(t+1))])
```

### How is this paper novel?

### Key takeaways

### What I still do not understand?

```
(C(t+1);C(t+1)h )   Cu([C(t)h ;M>Lmsg(L(t))])
(L(t+1);L(t+1)h )   Lu([L(t)h ; Flip(L(t));MCmsg(C(t+1))])
```

### Ideas to pursue

1. Explore speedups of using tensorflow-gpu for parallel brute force SAT solving compared to CPU.
