# [MIT 15.401 Finance Theory I, Fall 2008](https://www.youtube.com/playlist?list=PLUl4u3cNGP63B2lDhyKOsImI7FjCf6eDW)

## 1. Introduction

.

## 2. Present Value Relations 1

Asset
- sequence of future cashflows
- Anything that can be made into money
- Patents - assets out of ideas

Value of asset
`V(A) = (t[1]/t[0])V(A[1]) + ... + (t[n]/t[0])V(A[n])`
`V(A) = exchange_rate_vector . value_vector`

Impatience = The loss of value with time (should not be confusted with inflation)

Supply-Demand - More people want money today than tomorrow

## 3. Present Value Relations 2

Time value of money - Exponential decay

Perpetuity

`PV = C/(1+r) + C/(1+r)^2 + ...`
`PV = C / r`

Growth

`PV = C/(1+r) + C(1+g)/(1+r)^2 + ...`
`PV = C / (r - g), r > g` 

Annuity - PV after time T

Equivalent to purchasing a perpetuity and then selling it at `t+1`. **Indexing starts at 1**

`PV = C/r - (C/r) * (1+r)^-t`

Effective Annual Rate
EAR = (1+r/n)^n - 1
