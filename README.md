# Abstention Reranker

This repository contains the code for the ACL submission "Towards Trustworthy Reranking: A simple yet effective abstention mechanism".

## Abstract

Neural Information Retrieval (NIR) has significantly improved upon heuristic-based IR systems, yet failures remain frequent, as the models used sometimes do not manage to retrieve documents relevant to the user's query. We address these challenges by proposing a lightweight abstention mechanism tailored for real-world constraints, with particular emphasis placed on the reranking phase. We introduce a protocol for evaluating abstention strategies in a black-box scenario, demonstrating their efficacy, especially with access to reference data. We provide code for replicating experiments and implementing abstention mechanisms, fostering wider adoption and application in diverse contexts.

## Installation
```python
pip install -r requirements.txt
```

## Usage examples

See `/notebooks/implem.ipynb`.
