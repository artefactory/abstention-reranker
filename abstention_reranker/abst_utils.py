import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, ndcg_score

from abstention_reranker.utils import sort_scores


class AbstentionReranker:
    def __init__(self, method, metric, alpha=0.1, quantile_bad=0.25, quantile_good=0.75):
        self.method = method
        self.scorer = get_scorer(self.method, alpha)
        self.metric = metric
        self.alpha = alpha
        self.quantile_bad = quantile_bad
        self.quantile_good = quantile_good

    def evaluate_instances(self, relevance_scores_ref, targets_ref):
        self.relevance_scores_ref = relevance_scores_ref
        self.metrics_ref = evaluate_instances(relevance_scores_ref, targets_ref, self.metric)

    def fit_scorer(self, relevance_scores_ref=None, metrics_ref=None):

        if self.method not in ("max", "std", "1-2"):
            
	    if (relevance_scores_ref is not None) and (metrics_ref is not None):
	        self.relevance_scores_ref = relevance_scores_ref
	        self.metrics_ref = metrics_ref

            if self.method in ("linreg", "logreg"):
                sorted_scores_ref = np.sort(self.relevance_scores_ref, axis=1)
                #todo: add preprocessing for logreg
                self.scorer.fit(sorted_scores_ref)

	    elif self.method == "mahalanobis":
	        self.scorer = get_scorer_mah(
	            self.relevance_scores_ref, self.metrics_ref, self.quantile_bad, self.quantile_good
		)

    def compute_confidence_scores(self, relevance_scores):
        sorted_scores = np.sort(relevance_scores, axis=1)
        self.confidence_scores = self.scorer.predict(sorted_scores)
        return self.confidence_scores

    def rank_with_abstention(self, relevance_scores_test, target, target_type="abstention"):
        conf_scores_ref = self.compute_confidence_scores(self.relevance_scores_ref)
        conf_scores_test = self.compute_confidence_scores(relevance_scores_test)

        if target_type == "abstention":
            thold = np.quantile(conf_scores_ref, target)
            abst_decisions = conf_scores_test > thold

        elif target_type == "performance":
            metrics_ref_cummean = np.cumsum(np.sort(self.metrics_ref)[::-1]) / np.arange(1, len(self.metrics_ref) + 1)
            rate = (metrics_ref_cummean <= target).mean()
            abst_decisions = conf_scores_test > np.quantile(conf_scores_test, rate)

        return abst_decisions


def get_scorer(method, alpha):
    def scorer_1_2(relevance_scores):
        sorted_scores = np.sort(relevance_scores, axis=1)
        return sorted_scores[:, -1] - sorted_scores[:, -2]

    dict(
	"max": lambda relevance_scores: relevance_scores.max(axis=1),
        "std": lambda relevance_scores: relevance_scores.std(axis=1),
        "1-2": scorer_1_2,
        "linreg": Ridge(alpha=alpha),
        "logreg": TriclassLogreg(C=1 / alpha),
    )

    return scorer_by_name[name]


def get_scorer_linreg(relevance_scores_ref, metrics_ref, alpha=0.1, return_coefs=False):
    sorted_scores = np.sort(relevance_scores_ref, axis=1)
    reg.fit(sorted_scores, metrics_ref)


Class TriclassLogreg(LogisticRegresstion):

    def predict(self, scores):
        probas = super(self).predict_proba(scores)
        return probas[:, 2] - probas[:, 0]


def get_scorer_logreg(relevance_scores_ref, metrics_ref, alpha, quantile_bad, quantile_good, return_coefs=False):
    sorted_scores_ref = np.sort(relevance_scores_ref, axis=1)
    metric_classes = np.zeros_like(metrics_ref)
    metrics_argsort = np.argsort(metrics_ref)
    metric_classes[metrics_argsort[: round(len(metrics_ref) * quantile_bad)]] = -1
    metric_classes[metrics_argsort[round(len(metrics_ref) * quantile_good) :]] = 1

    clf.fit(sorted_scores_ref, metric_classes)


def get_scorer_mah(relevance_scores_ref, metrics_ref, quantile_bad, quantile_good):
    sorted_scores_ref = np.sort(relevance_scores_ref, axis=1)
    metrics_argsort = np.argsort(metrics_ref)
    relevance_scores_ref_bad = sorted_scores_ref[metrics_argsort[: round(len(metrics_ref) * quantile_bad)]]
    mah_bad = MahalanobisDistance()
    mah_bad.fit(relevance_scores_ref_bad)
    relevance_scores_ref_good = sorted_scores_ref[metrics_argsort[round(len(metrics_ref) * quantile_good) :]]
    mah_good = MahalanobisDistance()
    mah_good.fit(relevance_scores_ref_good)

    def scorer(relevance_scores):
        sorted_scores = np.sort(relevance_scores, axis=1)
        dist_to_bad = mah_bad.compute_distances(sorted_scores)
        dist_to_good = mah_good.compute_distances(sorted_scores)
        return dist_to_bad - dist_to_good

    return scorer


class MahalanobisDistance:
    def init(self):
        pass

    def fit(self, reference_set):
        self.ref_mean = reference_set.mean(axis=0)

        try:  # try to invert covariance matrix
            self.ref_inv_cov = np.linalg.inv(np.cov(reference_set, rowvar=False))
        except:  # otherwise return identity (Euclidian distance)
            self.ref_inv_cov = np.eye(reference_set.shape[1])

    def compute_distances(self, x):
        diff = x - self.ref_mean
        return np.array([np.sqrt(np.dot(np.dot(d, self.ref_inv_cov), d.T)) for d in diff])


def evaluate_instances(relevance_scores, targets, metric):
    if metric == "AP":
        metrics = compute_aps(relevance_scores, targets)

    elif metric == "NDCG":
        metrics = compute_ndcgs(relevance_scores, targets)

    elif metric == "RR":
        metrics = compute_rrs(relevance_scores, targets)

    return metrics


def compute_aps(scores, targets):
    num_instances = scores.shape[0]
    maps = np.zeros(num_instances)

    for i in range(num_instances):
        maps[i] = average_precision_score(targets[i], scores[i])

    return maps


def compute_ndcgs(scores, targets):
    num_instances = scores.shape[0]
    ndcgs = np.zeros(num_instances)

    for i in range(num_instances):
        ndcgs[i] = ndcg_score(targets[i].reshape(1, -1), scores[i].reshape(1, -1))

    return ndcgs


def compute_rrs(scores, targets):
    num_instances, num_docs = scores.shape[0], scores.shape[1]
    _, sorted_targets = sort_scores(scores, targets)
    rrs = np.zeros(num_instances)

    for i in range(num_instances):
        rrs[i] = 1 / (num_docs - np.where(sorted_targets[i] == 1)[0][-1])

    return rrs
