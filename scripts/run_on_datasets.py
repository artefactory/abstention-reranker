from abstention_reranker.ressources import BIENCODERS, XENCODERS, CUSTOM_XENCODERS, DATASETS, FRENCH_DATASETS, prefix_queries, prefix_docs
from abstention_reranker import load_reranking_dataset, process_dataset, compute_document_scores_xencoder, compute_document_scores, compute_document_scores_custom_xencoder
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle
import os


output_path = "./data/computed_scores/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

num_docs_pr = 10

for model_name in BIENCODERS + XENCODERS + CUSTOM_XENCODERS:
    if model_name in XENCODERS:
        model = CrossEncoder(model_name)
    elif model_name in BIENCODERS:
        model = SentenceTransformer(model_name)
    elif model_name in CUSTOM_XENCODERS:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        raise NotImplementedError

    for dataset_path in DATASETS + FRENCH_DATASETS[:1]:

        print("Computing scores for", model_name, dataset_path)
        save_path = output_path + model_name.replace("/", "_") + "_" + dataset_path.replace("/", "_") + ".pkl"

        if os.path.exists(save_path):
            print("Already exists, skipping")
            continue

        queries, positives, negatives = load_reranking_dataset(dataset_path)

        # try it until it runs
        i = 0
        while i < num_docs_pr:
            try:
                queries, positives_pr, negatives_pr = process_dataset(queries, positives, negatives, num_docs_pr - i, random_seed=42)
                break
            except:
                print("Failed, retrying with less docs", num_docs_pr - i)
                i += 1
                pass
        if i == num_docs_pr:
            print("Failed, skipping")
            continue

        if model_name in XENCODERS:
            scores, targets = compute_document_scores_xencoder(prefix_queries(queries, model_name, dataset_path),
                                                               prefix_docs(positives_pr, model_name),
                                                               prefix_docs(negatives_pr, model_name),
                                                               model=model)
        elif model_name in BIENCODERS:
            scores, targets = compute_document_scores(prefix_queries(queries, model_name, dataset_path),
                                                      prefix_docs(positives_pr, model_name),
                                                      prefix_docs(negatives_pr, model_name),
                                                      model=model)
        elif model_name in CUSTOM_XENCODERS:
            scores, targets = compute_document_scores_custom_xencoder(prefix_queries(queries, model_name, dataset_path),
                                                                      prefix_docs(positives_pr, model_name),
                                                                      prefix_docs(negatives_pr, model_name),
                                                                      model=model, tokenizer=tokenizer)
        else:
            raise NotImplementedError

        # save in dict
        scores_dict = {
            "model_name": model_name,
            "dataset_path": dataset_path,
            "num_docs_pr": num_docs_pr - i,
            "random_seed": 42,
            "queries": queries,
            "positives": positives_pr,
            "negatives": negatives_pr,
            "scores": scores,       # unsorted
            "targets": targets,     # unsorted
        }

        # save as pickle
        with open(save_path, "wb+") as f:
            pickle.dump(scores_dict, f)