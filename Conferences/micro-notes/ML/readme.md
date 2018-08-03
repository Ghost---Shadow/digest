# Machine Learning

## [Mukund Narasimhan, Engineer, Pinterest at MLconf Seattle 2017](https://www.youtube.com/watch?v=nl8a3DR2cXk)

1. Read more about word2vec and its uses in non NLP fields

## [John Maxwell, Data Scientist, Nordstrom at MLconf Seattle 2017](https://www.youtube.com/watch?v=3s8p2UjDF7c)

1. A/B testing is same as the multiarm bandit problem
2. Read more about inverse propensity scaling

## [Andrew Musselman, Committer & PMC Member, Apache Mahout at MLconf Seattle 2017](https://www.youtube.com/watch?v=Qkew4O3DfHA)

1. Checkout jCuda

## [Scott Clark, CEO & Co-Founder, SigOpt at MLconf Seattle 2017](https://www.youtube.com/watch?v=hopMOr7zsUQ)

1. Read about bayesian optimization

## [Machine learning - Introduction to Gaussian processes](https://www.youtube.com/watch?v=4vGiHC35j9s)

1. Fit a very high dimensional gaussian distribution. Slices of this distribution are the marginals.

## [Machine learning - Bayesian optimization and multi-armed bandits](https://www.youtube.com/watch?v=vz3D36VXefI)

1. Probability of improvement: Area under the bell curve when cut by a line which is the best outcome so far.

## [Gilles Louppe | Bayesian optimization with Scikit-Optimize](https://www.youtube.com/watch?v=DGJTEBt0d-s)

1. `pip install scikit-optimize`
2. Ask and tell API looks really easy to use.
3. Excellent for non-differentiable problems where evaluation is expensive.

## [Forecasting Long-Term Stock Returns](https://www.youtube.com/watch?v=u7Uv1uba8eg)

1. price/sales = *divide the per-share stock price by the per-share revenue*
2. price/book = (Shares/value of assets) *Price-to-book value (P/B) is the ratio of market value of a company's shares (share price) over its book value of equity. The book value of equity, in turn, is the value of a company's assets expressed on the balance sheet. This number is defined as the difference between the book value of assets and the book value of liabilities.*
3. Market capitalization = *market value of a publicly traded company's outstanding shares*
4. P/S and P/B are inversely correlated with returns (as expected)
5. Another way of thinking about this is, *For the same revenue, if more stocks are bought then the revenue will be divided among more people. Therefore the returns would be less.* 
6. Does 5 mean that revenue from stocks itself isnt a sustainable model?
7. Investigate outliers - Annualized returns to P/S, 7 year period, returns > 10 and 1.0 < P/S < 1.5 
8. Does it mean, if you own a company keep P/S and P/B as low as possible?
