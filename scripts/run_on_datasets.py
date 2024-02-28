from abstention_reranker.ressources import BIENCODERS, XENCODERS, CUSTOM_XENCODERS, MISTRAL_BIENCODERS , prefix_queries, prefix_docs
from abstention_reranker import load_reranking_dataset, process_dataset, compute_document_scores_xencoder, compute_document_mistral_scores, compute_document_scores, compute_document_scores_custom_xencoder
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import os
import argparse
import yaml
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--num_docs_pr", type=int, default=10)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--output_path", type=str, default="data_raw/")
parser.add_argument("--config_path", type=str, default="scripts/configs/run_config.yaml")
args = parser.parse_args()

output_path = args.output_path

# open config file
with open(args.config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    model_names = config["models"]
    dataset_paths = config["datasets"]

if not os.path.exists(output_path):
    os.makedirs(output_path)

num_docs_pr = args.num_docs_pr


for model_name in model_names:
    if model_name in XENCODERS:
        model = CrossEncoder(model_name)
    elif model_name in BIENCODERS:
        model = SentenceTransformer(model_name)
    elif model_name in CUSTOM_XENCODERS :
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name) #, device=torch.device("cuda"))
    elif model_name in MISTRAL_BIENCODERS:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, device_map="auto")
    else:
        print("Model not found", model_name)
        raise NotImplementedError

    for dataset_path in dataset_paths:

        print("Computing scores for", model_name, dataset_path)
        save_path = output_path + model_name.replace("/", "_") + "_" + dataset_path.replace("/", "_") + ".json"

        if os.path.exists(save_path):
            print("Already exists, skipping")
            continue

        queries, positives, negatives = load_reranking_dataset(dataset_path)

        # # try it until it runs
        # i = 0
        # while i < num_docs_pr:
        #     try:
        #         queries, positives, negatives = process_dataset(queries, positives, negatives, num_docs_pr - i, max_num_pos_pr=5, random_seed=args.random_seed)
        #         break
        #     except Exception as e:
        #         print("Failed, retrying with less docs", num_docs_pr - i)
        #         i += 1
        #         print(e)
        # if i == num_docs_pr:
        #     print("Failed, skipping")
        #     continue

        try:
            if model_name in XENCODERS:
                scores, targets = compute_document_scores_xencoder(prefix_queries(queries, model_name, dataset_path),
                                                                   prefix_docs(positives, model_name),
                                                                   prefix_docs(negatives, model_name),
                                                                   model=model)
            elif model_name in BIENCODERS:
                scores, targets = compute_document_scores(prefix_queries(queries, model_name, dataset_path),
                                                          prefix_docs(positives, model_name),
                                                          prefix_docs(negatives, model_name),
                                                          model=model)
            elif model_name in CUSTOM_XENCODERS:
                scores, targets = compute_document_scores_custom_xencoder(prefix_queries(queries, model_name, dataset_path),
                                                                          prefix_docs(positives, model_name),
                                                                          prefix_docs(negatives, model_name),
                                                                          model=model, tokenizer=tokenizer)
            elif model_name in MISTRAL_BIENCODERS:
                scores, targets = compute_document_mistral_scores(prefix_queries(queries, model_name, dataset_path),
                                                                  prefix_docs(positives, model_name),
                                                                  prefix_docs(negatives, model_name),
                                                                  model=model, tokenizer=tokenizer)
            else:
                raise NotImplementedError
        except Exception as e:
            print(f"Failed to compute scores for {model_name} on {dataset_path}, skipping")
            print(e)
            continue

        # save in dict
        scores_dict = {
            #"model_name": model_name,
            #"dataset_path": dataset_path,
            #"num_docs_pr": num_docs_pr - i,
            #"random_seed": 42,
            #"queries": queries,
            #"positives": positives,
            #"negatives": negatives,
            #"scores": scores.tolist(),       # unsorted
            #"targets": targets.tolist(),     # unsorted
            "scores": scores,
            "targets": targets,
        }

        # save as json
        with open(save_path, "w") as f:
            json.dump(scores_dict, f)
