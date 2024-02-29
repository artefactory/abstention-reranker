import numpy as np
import torch
from tqdm import tqdm


def compute_document_scores_custom_xencoder(queries, positives_pr, negatives_pr, model, tokenizer, device):
    model.eval()
    num_instances = len(queries)
    num_docs_pr = len(positives_pr[0]) + len(negatives_pr[0])
    model = model.to(device)
    #scores, targets = np.zeros((num_instances, num_docs_pr)), np.zeros((num_instances, num_docs_pr))
    scores, targets = [], []

    def encode_sample_xencoder(query, positive, negative):
        # model_name is for caching
        pairs = [[query, doc] for doc in (positive + negative)]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            scores_instance = (
                model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
                #.numpy()
                .tolist()
            )
        # scores_instance_argsort = np.argsort(scores_instance)
        #return scores_instance, np.array([1] * len(positive) + [0] * len(negative))
        return scores_instance, [1] * len(positive) + [0] * len(negative)

    for i, (query, positive, negative) in tqdm(list(enumerate(zip(queries, positives_pr, negatives_pr)))):
        #scores[i], targets[i] = encode_sample_xencoder(query, positive, negative)
        sc, tgt = encode_sample_xencoder(query, positive, negative)
        scores.append(sc)
        targets.append(tgt)

    return scores, targets
