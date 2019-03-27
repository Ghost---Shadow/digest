## Key takeaways

1. Sub pixel accuracy for tracking features can be found by finding the double derivative of the image.

2. LSM = minimize(last_image(i, j) - current_image(i - shift_x, j - shift_y))

g = observed_img
f = last_img

f(x + du, y + dv) ~= f(x,y) + grad(f,u) du + grad(f,v) dv

Reshaping to 1D

g[m] - noise[m] = f[m] + grad(f,x)[m] * du + grad(f,y)[m] dv

Solve for du and dv using numerical methods

vec(g) - vec(f) - vec(n) = Jacobean(f) * vec([du, dv])

A = Jacobean(f)
dl = vec(g) - vec(f)
v = vec(n)
dx = vec([du, dv])

dl - v = A * dx
v = 0?
dl  = A * dx
A.T * A * dx = A.T * dl
dx = ((A.T * A) ^ -1) * A.T * dl

More gradients are, the better matching I get

cov(A.T * A) = 1/m * A.T * A

cov(?,?) = cov(n,n) / m * (A.T * A) ^ -1

Intuition (A.T * A) ^ -1: Probabilistic shift direction of the kernel for maximum overlap


