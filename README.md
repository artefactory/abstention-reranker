# Abstention Reranker

Reference article: "[Towards Trustworthy Reranking: A Simple yet Effective Abstention Mechanism](https://arxiv.org/pdf/2402.12997.pdf)" (accepted at TMLR, 09/2024).

Test

## Abstract

Neural Information Retrieval (NIR) has significantly improved upon heuristic-based Information Retrieval (IR) systems. Yet, failures remain frequent, the models used often being unable to retrieve documents relevant to the user's query. We address this challenge by proposing a lightweight abstention mechanism tailored for real-world constraints, with particular emphasis placed on the reranking phase. We introduce a protocol for evaluating abstention strategies in black-box scenarios (typically encountered when relying on API services), demonstrating their efficacy, and propose a simple yet effective data-driven mechanism. We provide open-source code for experiment replication and abstention implementation, fostering wider adoption and application in diverse contexts.

## Installation
```python
pip install -r requirements.txt
```

## Computation of relevance scores

```python 
python scripts/run_on_datasets.py --config-path <path to config YML>
```

## Experiment replication

See [/notebooks/plots.ipynb](https://github.com/artefactory/abstention-reranker/blob/main/notebooks/plots.ipynb).

## Usage examples

See [/notebooks/implem.ipynb](https://github.com/artefactory/abstention-reranker/blob/main/notebooks/implem.ipynb).

## Reference

If you found our work useful, please consider citing:

```
@misc{gisserotboukhlef2024trustworthy,
    title={Towards Trustworthy Reranking: A Simple yet Effective Abstention Mechanism}, 
    author={Hippolyte Gisserot-Boukhlef and Manuel Faysse and Emmanuel Malherbe and Céline Hudelot and Pierre Colombo},
    year={2024},
    eprint={2402.12997},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
```
