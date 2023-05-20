# [MIT 6.854 Advanced Algorithms](https://www.youtube.com/playlist?list=PL6ogFv-ieghdoGKGg2Bik3Gl1glBTEu8c)

## Lecture 13: Submodular Functions

### A function is submodular if

```txt
f(A U {j}) - f(A) >= f(B U {j}) - f(B)
```

A is a subset of B. j does not belong to B

```txt
f(A int B) + f(A U B) <= f(A) + f(B)
```

### A function is convex if

g is the function in question
x and y are two points
t is [0,1]

Function is convex if the following is true for all x,y

```txt
t * g(x) + (1 - t) * g(y) >= g(t * x + (1 - t) * y)
```

### Lovaz extension

Given a solid hypercube cube, each vertex is 1 hot representation of a set element

```txt
i = ?
z = ?
t = ?
f(z) = E(f({i,z>=t}))
```

Assumptions

```txt
1. f(phi) = 0
2. z1 >= z2 >= ... >= zn => Si = first i elements of zi
3. f(z) = sum((z_i+1 - z_i) * F(S_i)) + z_n * F(S_n)
```

f is convex iff f is submodular

#### Proof

```txt

max ZTx
```
