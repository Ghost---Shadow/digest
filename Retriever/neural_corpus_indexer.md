# [A Neural Corpus Indexer for Document Retrieval](https://arxiv.org/abs/2206.02743)

**Source Code:** https://github.com/solidsea98/Neural-Corpus-Indexer-NCI

**Datasets:** NQ320k, TriviaQA

**Author:** Microsoft

**Journal:** Neurips (Recipeient of outstanding paper award)

**Year of Submission:** 2022

## What problem does it solve?

## How does it solve it?

### Training flow

#### Query generator

1. Given the supporting documents, use the generative model to produce a question.
2. Calculate the Cross-Entropy (CE) loss comparing the generated question to the actual dataset question.

#### Retriever

1. Load dataset in the form of supporting_document, question and answer
2. All supporting documents are semantically embedded and then Perform k-means clustering on these embeddings.
3. Depending on a document's position in the k-means clustering hierarchy, generate a docid that encapsulates semantic information about the document's location in the k-means tree.
4. To handle tokens that have different meanings based on their position, the decoder takes both position and token value as inputs. So, "315253" becomes “(1,3)(2,5)(3,5)”.
5. Train a network to predict doc_id given a question (the question can also be a generated question)
6. An auxilary network uses the position information and generates a weight matrix which is multiplied to the pre-softmax activation in the primary network.
7. Also see the Lreg in equations section

#### Answer generator

1. Using the supporting documents and the generated question, produce an answer.
2. Calculate the Cross-Entropy (CE) loss comparing the generated answer to the actual dataset answer.

### Document ID generation

Hierarchical k-means Algorithm: To achieve the above goal, the hierarchical k-means algorithm is employed to encode the documents.
    Every document in the collection is initially categorized into 'k' clusters based on their BERT-encoded representations.
    For clusters with more than 'c' documents, the k-means algorithm is reapplied in a recursive manner.
    Clusters containing 'c' or fewer documents get an assigned number for each document, ranging from 0 to c-1.

Tree Structure and Semantic Identifiers:
    The documents are organized into a tree structure, T, with root r0.
    Every document is tied to a leaf node and has a unique path l=r0,r1,...,rml=r0,r1,...,rm leading from the root to that leaf. Here, ri∈[0,k)ri∈[0,k) denotes the cluster index for level 'i', and rm∈[0,c)rm∈[0,c) represents the leaf node.
    The semantic identifier of a document is a concatenation of node indices spanning from the root to its leaf node.
    If documents are semantically similar, the beginning portions (prefixes) of their identifiers are likely identical.

Experiment Parameters: In the experiments, both 'k' and 'c' were set to 30. Optimizing these hyper-parameters remains a task for future research.

### Inference

1. For incoming question use the doc_id generator model to generate N doc_ids using beam search
2. Lookup the doc_ids in database to get the documents
3. Take the documents + question to get answer

### Equations

Lreg = -log( exp(sim(zi,1, zi,2)/τ) / Σ(from k=1 to 2Q, k!=2) exp(sim(zi,1, zi,k)/τ) )

- Lreg doesnt seem to be used in code?
- range of k is 1,2 it cannot go upto 2Q?

query_tloss=CE(L,Q×D)query_tloss=CE(L,Q×D)

### Model

semantic embeddings generation = bert-base-uncased

query generation = doc2query-t5-base-msmarco

document id generation (retrieval model) = t5-small, t5-base, t5-large

## How is this paper novel?

Embed a semantically readable document_id with each supporting document. This informs the network where that document came from.

## List of experiments

Metrics - Recall@N, Mean Reciprocal Rank (MRR), R-Precision

### Ablation Studies

- Without DocT5Query: Removing generated training queries by DocT5Query showed a significant performance boost, corroborating the hypothesis that training with augmented queries aids the NCI model in grasping the document semantics better.
- Without document as query: Using document content as queries aids the model in understanding document semantics, similar to the DSI model.
- Without PAWA decoder: Eliminating the adaptive decoder layer showed that a tailored decoder architecture specifically designed for semantic identifier generation is essential.
- Without semantic id: Swapping the semantic identifier with a randomly generated one showed a decline in model performance. This hints that the semantic identifiers obtained through hierarchical k-means are crucial.
- Without regularization: Absence of consistency-based regularization loss leads to performance decline, as the decoder is susceptible to over-fitting.
- Without constrained beam search: Disabling the validating constraint in beam search deteriorates performance across evaluation metrics, emphasizing the value of prior structures in beam search.

NOTE:

- recall@1-triviaQA minimum score is 60 max is 65
- recall@5-nq minimum score is 84 max is 90

The ablations help, but none are dealbreaker

### Efficiency analysis

beamsize, latency, throughput

seems fine

## Preliminaries

## GPU hours

8 x NVIDIA V100 GPU with 32GB

## Key takeaways

## What I still do not understand?

## Ideas to pursue

## Similar papers
