import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def compute_document_scores(queries_pr, positives_pr, negatives_pr, model):

    num_instances = len(queries_pr)
    num_docs_pr = len(positives_pr[0]) + len(negatives_pr[0])
    scores, targets = np.zeros((num_instances, num_docs_pr)), np.zeros((num_instances, num_docs_pr))

    def encode_sample(query, positive, negative):
        # model_name is for caching
        query_emb = model.encode(query).reshape(1, -1)
        doc_emb = model.encode([doc for doc in (positive + negative)], normalize_embeddings=True)
        # compute dot score and cast to numpy
        scores_instance = util.cos_sim(query_emb, doc_emb)[0].numpy()

        return scores_instance, np.array([1] * len(positive) + [0] * len(negative))

    for i, (query, positive, negative) in tqdm(list(enumerate(zip(queries_pr, positives_pr, negatives_pr)))):
        scores[i], targets[i] = encode_sample(query, positive, negative)

    return scores, targets
