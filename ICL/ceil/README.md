# [Compositional Exemplars for In-context Learning](https://arxiv.org/abs/2302.05698)

## Meta

* Journal - ICML
* Year - 2023
* Author - Hong Kong University
* Code - https://github.com/HKUNLP/icl-ceil
* One liner - We optimize the embedding space for retrieval in such a way that, given a question, the volume spanned by embeddings of relevant documents is larger than non-relevant documents.
* Model - GPT-Neo-2.7B, bert-base-uncased
* Datasets - WebQs, GeoQuery, NL2Bash, MTOP, and SMCalFlow, LFEM
* Baselines - Random, BM25-topk, BERT-topk, [EPR](https://aclanthology.org/2022.naacl-main.191.pdf)

## Training flow

1. For an incoming question, create and embedding. Find n most similar documents from the database N.
2. Take each document and question and check the likelihood of answering the question using a generating LLM.
3. Sort the documents by this score and divide into two parts. S+ and S-. S+ being highest scoring half and S- being lowest scoring half.
4. Use the contrastive loss to train the embedding model. It enforces both quality and diversity.

## Inference flow

1. For an incoming question, create and embedding. Find n most similar documents from the database N.
2. Add the top example to S_map
3. For each of the remaining documents (i) compute log(L'(S_map U {i})) - log(L'(S_map)) this is the likelihood gain by adding ith document
4. Put the best document in S_map
5. Repeat until S_map is of target size

## Equations

~k(ai, aj | x) = g(ai, x) k(ai, aj) g(aj , x)

- ai, aj, x are semantic embeddings
- g and k are dot products
- ~k can be thought of a metric which is high when retrived documents are relevant and similar

P(S) = det(L_S) / det(L+I)

- S is the retrieved documents
- L is the similarity matrix of the entire corpus
- L_S is the similarity matrix of the retrieved documents
- det(L) represents the volume of the parallelepiped spanned by the points of the similarity matrix
- If the volume of the retrieved documents increase then we can say that they are more diverse

Also read Gram Matrix

k~(ai,aj∣x)=g(ai,x)k(ai,aj)g(aj,x)

- k~(ai,aj∣x) computes similarity of ai and aj given x
- g is relevance score (dot product)
- k is similarity function (dot product)
- Increasing diversity is the responsibility of P(S)

logdet(L~S​)=∑i∈S​log(g(ai,x) ** p​)+logdet(LS​)

- The first term measures similarity between incoming question and the retrieval set
- The second term measures diversity between the retrieval set elements
- Both together measure quality and diversity
- p = 2, hyperparameter

logP(S−)−logP(S+)
= log(det(LS−​)/det(L+I))​−log(det(LS+​)/det(L+I)​)
= (logdet(LS−​) - det(L+I)) − (logdet(LS+​) - det(L+I))
= logdet(LS−​)−logdet(LS+​)

j=arg maxi∈Z∖Smap​​[logdet(LSmap​∪{i}′​)−logdet(LSmap​′​)]

## Experiments

- Downstream LLM accuracy
- Inferencer transfer
- different contrastive loss ablations
- sampling strategy ablations
- number of in context examples
- inference latency vs num shots

## Proofs

## Key takeaways

- Using a smaller generator model for training and then using larger for inference can cut down costs

## Similar papers

- [Diverse Multi-Answer Retrieval with Determinantal Point Processes](https://arxiv.org/abs/2211.16029)
