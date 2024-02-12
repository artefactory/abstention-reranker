import numpy as np
from tqdm import tqdm


def compute_document_scores_xencoder(queries, positives_pr, negatives_pr, model):

    num_instances = len(queries)
    num_docs_pr = len(positives_pr[0]) + len(negatives_pr[0])
    scores, targets = np.zeros((num_instances, num_docs_pr)), np.zeros((num_instances, num_docs_pr))

    def encode_sample_xencoder(query, positive, negative):
        # model_name is for caching
        scores_instance = model.predict([[query, doc] for doc in (positive + negative)], apply_softmax=True)
        # scores_instance_argsort = np.argsort(scores_instance)
        # if multiple heads - assume head 0 is the one we want
        if len(scores_instance.shape) > 1 and scores_instance.shape[1] > 1:
            scores_instance = scores_instance[:, 0]
        return scores_instance, np.array([1] * len(positive) + [0] * len(negative))

    for i, (query, positive, negative) in tqdm(list(enumerate(zip(queries, positives_pr, negatives_pr)))):
        scores[i], targets[i] = encode_sample_xencoder(query, positive, negative)

    return scores, targets
