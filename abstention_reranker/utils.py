import json
import os

import numpy as np
from datasets import load_dataset


def load_reranking_dataset(path):
    dataset = load_dataset(path)

    try:
        queries, positives, negatives = (
            dataset["test"]["query"],
            dataset["test"]["positive"],
            dataset["test"]["negative"],
        )
    except:
        queries, positives, negatives = dataset["dev"]["query"], dataset["dev"]["positive"], dataset["dev"]["negative"]

    return queries, positives, negatives


def process_dataset(queries, positives, negatives, num_docs_pr=10, max_num_pos_pr=5, random_seed=42):
    # set random seed
    np.random.seed(seed=random_seed)

    queries_pr, positives_pr, negatives_pr = [], [], []

    for query, positive, negative in zip(queries, positives, negatives):
        positive, negative = list(set(positive)), list(set(negative))  # remove duplicate docs
        num_docs_instance = len(positive + negative)

        # sample docs
        if (
            len(set(positive) & set(negative)) == 0
        ):  # check if there is an intersection between positive and negative docs
            if num_docs_instance >= num_docs_pr:  # check if more docs in instance than target number of docs
                num_sampled_pos = min(max_num_pos_pr, len(positive))

                if len(negative) >= num_docs_pr - num_sampled_pos:  # check if enough negative docs to sample
                    sampled_pos = np.random.choice(positive, num_sampled_pos, replace=False).tolist()
                    sampled_neg = np.random.choice(negative, num_docs_pr - num_sampled_pos, replace=False).tolist()
                    positives_pr.append(sampled_pos)
                    negatives_pr.append(sampled_neg)
                    queries_pr.append(query)

    return queries_pr, positives_pr, negatives_pr


def load_relevance_scores_datasets_from_hf(hf_path, dump_path):
    dataset = load_dataset(hf_path)
    dataset_names_dict = {
        "scidocs-reranking": "SciDocs",
        "askubuntudupquestions-reranking": "AskUbuntu",
        "stackoverflowdupquestions-reranking": "StackOverflow",
        "mteb-fr-reranking-alloprof-s2p": "Alloprof",
        "CMedQAv1-reranking": "CMedQAv1",
        "Mmarco-reranking": "Mmarco",
    }

    # create dump path
    if not os.path.exists(dump_path):
        os.makedirs(dump_path, exist_ok=True)

    for data in dataset["test"]:
        model_name = data["model_name"].split("/")[-1]
        dataset_path = data["dataset_path"].split("/")[-1]

        if dataset_path not in dataset_names_dict.keys():
            continue

        if model_name != "BAAI/bge-base-en":
            scores = np.array(data["scores"])
            targets = np.array(data["targets"])

            with open(os.path.join(dump_path, f"{dataset_names_dict[dataset_path]}_{model_name}.json"), "w") as json_file:
                json.dump({"scores": scores.tolist(), "targets": targets.tolist()}, json_file)


def load_relevance_scores_datasets_from_local(model_names, dataset_names, path):
    all_data = {}

    for model_name in model_names:
        all_data[model_name] = {}

        for i, dataset_name in enumerate(dataset_names):
            with open(os.path.join(path, f"{dataset_name}_{model_name}.json"), "r") as json_file:
                scores_dataset_model = json.load(json_file)

            all_data[model_name][dataset_names[i]] = {
                "scores": np.array(scores_dataset_model["scores"]),
                "targets": np.array(scores_dataset_model["targets"]),
            }

    return all_data


def sort_scores(scores, targets):
    sorted_indices = np.argsort(scores, axis=1)
    sorted_scores = scores[np.arange(scores.shape[0])[:, None], sorted_indices]
    sorted_targets = targets[np.arange(targets.shape[0])[:, None], sorted_indices]
    return sorted_scores, sorted_targets
