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

## 4. Present Value Relations 3

Mark to market - Establish a market price of an asset by querrying the market.

Increase in cost of living, I is inflation
```
= I[t + k]/I[t] 
= (1 + p) ^ k
```

Real wealth after k years 
`= W[t + k] / (1 + p) ^ k`

Real return
```
= W[t + k]/(W[t] * (1 + p) ^ k)
= (1 + r_nominal) / (1 + p) - 1
= r_nominal - p
r_real = r_nominal - p
```

Equity = Assets - Liabilities

Asset = Something you have
Liabilities = Something you owe

Futures = Financial contracts obligating the buyer to purchase an asset or the seller to sell an asset, such as a physical commodity or a financial instrument, at a predetermined future date and price.

Liquidity decreases with complexity of figuring out the market value of an asset.

Loan = money to borrow
Margin = A collateral the borrower has to give to the lender (broker)
Leverage = Purchasing more stocks than what the portfolio holder has, where the borrowable amount is scaled by the amount of money the holder has.

Derivative = Any asset whose value depends on the underlying set of assets is called a derivative

Coupon bonds = A bond which gives the bearer `x%` of its value every year (similar to dividend).

## 5. Fixed income securities 2

Investment grade assets = ?

`R[t]` = rate of interest at year `t` (Spot rate)

```
P = F / ((1 + R[1]) (1 + R[2]) ... (1 + R[T]))

let r = geometric_mean(1 + R) - 1
P = F / (1 + r) ^ T

P[t - 1] / P[t] = 1 + R[t]
R[t] is called forward rate
```

Coupon bonds

```
P = F/(1+R[1]) +  F/((1+R[1])(1+R[2])) +  F/((1 + R[1]) (1 + R[2]) ... (1 + R[T])))

y = r

r needs to be determined numerically
```

Expectation hypothesis: Expected Future Spot
```
E[0](R [k]) = f[k]
```
## 6. Fixed income securities 3

Yield curve = graph of interest rate vs maturity date (borrowing period)

Arbitrage = free lunch

Macaulay duration = Duration the investor is exposed to risk

Second order derivative of yield curve is volatility
