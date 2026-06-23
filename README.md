# SBERT Triplet Loss Fine-Tuning for Educational Semantic Retrieval

## Overview

This project investigates the effectiveness of Triplet Loss fine-tuning for improving Sentence-BERT (SBERT) representations in a domain-specific educational retrieval task. The study focuses on adapting pretrained sentence embeddings to a small-scale educational marketing dataset and evaluating their impact on semantic retrieval performance.

## Objectives

* Improve semantic retrieval performance through metric learning.
* Fine-tune SBERT using Triplet Loss.
* Evaluate different negative sampling strategies.
* Compare fine-tuned models against baseline retrieval approaches.
* Assess retrieval effectiveness using standard Information Retrieval metrics.

## Methodology

### Base Model

* Sentence-BERT (all-MiniLM-L6-v2)
* 384-dimensional sentence embeddings

### Fine-Tuning Strategy

Triplet Loss was applied to optimize embedding representations by bringing semantically related query-answer pairs closer while pushing unrelated pairs apart.

### Negative Sampling

The following strategies were evaluated:

* Random Negatives
* Semi-Hard Negatives
* Hard Negatives
* Mixed Negatives

### Retrieval Pipeline

1. Generate sentence embeddings using SBERT.
2. Fine-tune embeddings with Triplet Loss.
3. Index answer embeddings using FAISS.
4. Retrieve top-k candidates based on cosine similarity.
5. Evaluate retrieval effectiveness.

## Evaluation Metrics

* Precision@K
* Recall@K
* Mean Reciprocal Rank (MRR)
* nDCG@10
* Accuracy

## Key Results

Best-performing configuration:

* Learning Rate: 5e-5
* Batch Size: 16
* Epochs: 4
* Mixed Negative Sampling

Performance:

| Metric    | Score  |
| --------- | ------ |
| Recall@10 | 0.9750 |
| MRR       | 0.9302 |
| nDCG@10   | 0.9415 |
| Accuracy  | 97.5%  |

## Technologies

* Python
* Sentence Transformers
* PyTorch
* FAISS
* NumPy
* Pandas
* Scikit-learn

## Project Structure

```text
├── train.py
├── evaluate.py
├── retrieve.py
├── requirements.txt
├── data/
├── models/
├── results/
└── README.md
```

## Research Contributions

* Demonstrates the effectiveness of Triplet Loss for domain adaptation in low-resource educational datasets.
* Provides a comparative analysis of negative sampling strategies.
* Shows how dense retrieval methods can significantly outperform baseline sentence embedding approaches in domain-specific retrieval tasks.

## Future Work

* Retrieval-Augmented Generation (RAG)
* Neural-symbolic retrieval methods
* Multilingual semantic retrieval
* Large Language Model integration
* Explainable retrieval systems

## Author

Nurawati Mina

Master of Artificial Intelligence
Universitas Pelita Harapan
