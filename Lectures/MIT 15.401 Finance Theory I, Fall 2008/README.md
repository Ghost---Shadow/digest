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

## 7. Fixed income securities 4

**Securitization** - Polarising the risks of multiple *uncorrelated* bonds so that we can entice investors with both high risk and low risk appetite. 

Let there be two bonds, 1000 each, with 10% default risk.

The success matrix

| Event | Chance | Payout |
|-------|---------|-----------|
| Both succeed | 81% | 2000 |
| One succeeds | 18% | 1000 |
| Both fails   | 1% | 0 |

We as bond issuers issue both bonds but we have a special policy. For a premium, the investor can buy a hypothetical bond. Such that if one of the bonds, succeed, then he gets a payout. Or, he can buy another policy where he gets paid only if both bonds succeed. What should be the price of these two hypothetical bonds?

Senior Tranche = Probability of payout * payout = 99% * 1000 = 990
Junior Tranche = 81% * 1000 = 810

For perfectly correlated bonds

| Event | Chance | Payout |
|-------|---------|-----------|
| Both succeed | 81% | 2000 |
| Both fails   | 19% | 0 |

## 8. Equities

Equity = Part ownership in a corporation

Limited Liability = The most you can loose is the money you put in.

Short sale = Selling your asset before you default

Primary market = Venture capital, IPO

Secondary market = Stock exchanges

```
P[t] = Price of stock at t
D[t] = Cash dividend at t
E[t]() = Expectation operator (forecast) at t
r[t] = Risk adjusted discount rate for cashflow at t

P[t] = V[t](D[t + 1], D[t + 2], ...)
= E[t](D[t + 1]) / (1 + r[t + 1]) + E[t](D[t + 2]) / (1 + r[t + 2])^2 + ...

P[t] = sum(E[t](D[t + k]) / (1 + r[t + k]) ^ k, k = 1 to inf)
```

Assuming

```
D[t] = D
r[t] = r

P[inf] = P = D/r

with growth
P = D/(r - g)
```

## 9. Forward and Futures Contracts 1

**Forward/Futures contract** = A legal binding contract that a transaction will take place at some time in future.

Forwards contract only works when both buyer and seller think neither are losing value. Therefore the contract itself has no value.

Forwards are used to eliminate uncertainity. Lock in a price.

Forwards are useful when you want to buy something now but dont want to store it physically right now.

Futures = Every sub period of market fluctuations the parties pay the deviation from the market price and the agreed value. In this way they mitigate counterparty risk.
