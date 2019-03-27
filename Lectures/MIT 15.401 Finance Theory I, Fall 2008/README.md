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

## 10. Forward and Futures Contracts 2 and Options 1

Futures/Forwards price at time t = Buying the same amount at t=0 and then paying for storage for time t

```
F[t,T] / (1 + t) ^ (T - t) = P[t] + S[t]
```

Since stocks are very correlated, you can short sell the SNP500 and achieve the same effect as selling your portfolio.

Options = Futures but you can cancel the deal at any time

Put = Selling Options
Call = Buying Options

Put options are insurances but unlike insurances you can cash in any time. Also, they can be bought and sold on the exchange.

Early excercise = Early selling

## 11. Options 2

### Call option

Payoff diagram: Payoff vs stock price

Payoff diagram at maturity date. Dotted line is premium for the option.
![Payoff diagram](./11_payoff_diagram_call.png)

If the value of stock price `< 20$` at maturity then the payoff is zero.

The value of a option before maturity is non zero even if the stock price `< 20$` due to its potential.

### Put option

Sell the stock before

![Payoff diagram](./11_payoff_diagram_put.png)

Max payoff is bounded `10$`

Insurances are put options

### Option Strategies

#### Call + Put

![Payoff diagram](./11_payoff_diagram_call_and_put.png)

#### Call - Call

![Payoff diagram](./11_payoff_diagram_call_and_call.png)

Options price and volatility are related

### Corporate liabilities

V = ?
B = ?

Equity = V - B + Max (0, B - V)

Debt = B - Max(0, B - V)

V = Debt + Equity

### Other

Labor markets with fixed tenure look like options: After the fixed amount of money for the tenure, there are no more liabilities. 

Education is an option: After the payment of the education fees, there are no more liabilities.

## 12. Options 3 and Risk and Return 1

`S[0]` = Stock price right now

K = Strike price

Option expires tomorrow 
`C[1] = max(s[1] - K, 0)`

Suppose `S[0] -> S[1]` is Bernoulli trial

![Valuation of options](./12_valuation_of_options_1.png)

B = Bonds
r = Growth rate (e.g. 1.03)

`V[0] == S[0]\delta + B`

```
V[1]
= uS[0]\delta + rB = C[u] (with probability p)
or
= dS[0]\delta + rB = C[d] (1 - p)
```

![Valuation of options](./12_valuation_of_options_2.png)

Notice: There is no p in the formula

Buyer and seller have two different p but can agree on the price of the option.

```
d < r < u
```

If r < d then nobody would ever buy B because the rate of riskless borrowing is less than the lowest payout of riskful borrowing.

If r > u then the risk free borrowing is better than the riskful borrowing.

![Valuation of options](./12_valuation_of_options_3.png)

### Risk and Return

```
Return = R[i][t] = (D[i][t] + P[i][t] - P[i][t-1]) /  P[i][t-1]

Expected Return = E(R[i][t])

Excess Return = R[i][t] - r[f]

Risk Premium = E(R[i][t]) - r[f]
```
