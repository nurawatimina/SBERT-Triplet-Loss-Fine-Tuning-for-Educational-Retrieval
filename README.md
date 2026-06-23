# Project Overview
SBERT Triplet Loss Semantic Retrieval
This project investigates the effectiveness of Triplet Loss fine-tuning for Sentence-BERT in a domain-specific educational retrieval task.
# Research Objective
- Improve semantic retrieval performance
- Compare negative sampling strategies
- Evaluate retrieval effectiveness
# Methodology
Model: all-MiniLM-L6-v2

Loss:
- Triplet Loss

Negative Sampling:
- Random
- Semi-Hard
- Hard
- Mixed

# Result
| Metric    | Score  |
| --------- | ------ |
| Recall@10 | 0.975  |
| MRR       | 0.9302 |
| nDCG@10   | 0.9415 |
| Accuracy  | 97.5%  |
