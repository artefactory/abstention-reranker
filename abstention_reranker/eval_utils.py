import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from abstention_reranker.abst_utils import (AbstentionReranker,
                                            evaluate_instances,
                                            get_scorer_linreg)
from abstention_reranker.utils import load_reranking_dataset, process_dataset


def evaluate_oracle(metrics, abstention_rates):
    metrics_argsort = np.argsort(metrics)
    oracle_eval = np.zeros(len(abstention_rates))

    for i, rate in enumerate(abstention_rates):
        oracle_eval[i] = metrics[metrics_argsort[round(rate * len(metrics)) :]].mean()

    return oracle_eval


def evaluate_strategy(confidence_scores, metrics, abstention_rates):
    strat_eval = np.zeros(len(abstention_rates))
    abstention_thresholds = np.quantile(confidence_scores, abstention_rates)

    for i, abstention_threshold in enumerate(abstention_thresholds):
        strat_eval[i] = metrics[confidence_scores >= abstention_threshold].mean()

    return strat_eval


def evaluate_strategies_on_benchmark(
    all_data,
    abstention_rates,
    dataset_names,
    model_names,
    metric_names,
    methods,
    random_seeds,
    test_size=0.2,
    alpha=0.1,
    quantile_bad=0.1,
    quantile_good=0.9,
    quantile_score=0,
    ref_subsample_size=None,
):
    strat_evals = {
        model_name: {
            dataset_name: {
                metric: {
                    method: np.zeros((len(random_seeds), len(abstention_rates)))
                    for method in ["oracle", "random"] + methods
                }
                for metric in metric_names
            }
            for dataset_name in dataset_names
        }
        for model_name in model_names
    }

    for i, model_name in enumerate(model_names):
        print(f'[{i+1}/{len(model_names)}] {model_name}:')

        for j, dataset_name in enumerate(dataset_names):
            print(f'   - [{j+1}/{len(dataset_names)}] {dataset_name}:')
            rel_scores = all_data[model_name][dataset_name]["scores"]  # retrieve relevance scores
            targets = all_data[model_name][dataset_name]["targets"]  # retrieve targets

            for k, metric in enumerate(metric_names):
                print(f'      > [{k+1}/{len(metric_names)}] {metric}:')
                metrics = evaluate_instances(rel_scores, targets, metric)  # compute metrics

                for l, seed in tqdm(list(enumerate(random_seeds))):  # multiple runs (different seeds)
                    rel_scores_ref, rel_scores_test, metrics_ref, metrics_test = train_test_split(
                        rel_scores, metrics, test_size=test_size, random_state=seed
                    )  # ref/test split

                    if ref_subsample_size is not None:  # subsample ref set when relevant
                        np.random.seed(seed)
                        sampled_rows = np.random.choice(
                            rel_scores_ref.shape[0],
                            size=max(1, round(rel_scores_ref.shape[0] * ref_subsample_size)),
                            replace=False,
                        )
                        rel_scores_ref = rel_scores_ref[sampled_rows]
                        metrics_ref = metrics_ref[sampled_rows]

                    oracle_eval = evaluate_oracle(metrics_test, abstention_rates)  # evaluate oracle
                    strat_evals[model_name][dataset_name][metric]["oracle"][l] = oracle_eval
                    strat_evals[model_name][dataset_name][metric]["random"][l] = oracle_eval[0]

                    for method in methods:  # evaluate methods
                        conf_scorer = AbstentionReranker(
                            method, metric, alpha, quantile_bad, quantile_good, quantile_score
                        )  # initialize confidence scorer
                        conf_scorer.get_scorer(rel_scores_ref, metrics_ref)  # fit scorer (when relevant)
                        conf_scores_test = conf_scorer.compute_confidence_scores(
                            rel_scores_test
                        )  # compute confidence scores
                        strat_eval = evaluate_strategy(
                            conf_scores_test, metrics_test, abstention_rates
                        )  # evaluate strategy
                        strat_evals[model_name][dataset_name][metric][method][l] = strat_eval

    return strat_evals


def compute_naucs(strategy_evaluations, abstention_rates, dataset_names, model_names, metric_names, methods):
    naucs_df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [(d, m) for d in dataset_names for m in metric_names], names=["Dataset", "Metric"]
        ),
        columns=pd.MultiIndex.from_tuples(
            [(mo, me) for mo in model_names for me in methods], names=["Model", "Method"]
        ),
    )

    for model_name in model_names:
        for dataset_name in dataset_names:
            for metric in metric_names:
                auc_oracle = auc(
                    abstention_rates, strategy_evaluations[model_name][dataset_name][metric]["oracle"].mean(axis=0)
                )
                auc_random = auc(
                    abstention_rates, strategy_evaluations[model_name][dataset_name][metric]["random"].mean(axis=0)
                )

                for method in methods:
                    avg_perfs = strategy_evaluations[model_name][dataset_name][metric][method].mean(axis=0)
                    naucs_df.loc[(dataset_name, metric), (model_name, method)] = (
                        auc(abstention_rates, avg_perfs) - auc_random
                    ) / (auc_oracle - auc_random)

    return naucs_df


def evaluate_raw_performance_on_benchmark(all_data, dataset_names, model_names, metric_names):
    raw_perf_df = pd.DataFrame(
        index=model_names,
        columns=pd.MultiIndex.from_tuples(
            [(dataset_name, metric) for dataset_name in dataset_names for metric in metric_names],
            names=["Dataset", "Metric"],
        ),
    )
    raw_perf_df.index.name = "Model"

    for model_name in model_names:
        for dataset_name in dataset_names:
            rel_scores, targets = all_data[model_name][dataset_name].values()  # retrieve relevance scores and targets

            for metric in metric_names:
                raw_perf_df.loc[model_name, (dataset_name, metric)] = evaluate_instances(
                    rel_scores, targets, metric
                ).mean()  # evaluate instances

    return raw_perf_df


def compute_abstention_raw_performance_correlations(raw_perfs, naucs):
    dataset_names = raw_perfs.columns.get_level_values(0).unique()
    corr_df = pd.DataFrame(index=dataset_names, columns=["Correlation"])

    for dname in dataset_names:
        corr_df.loc[dname, "Correlation"] = pearsonr(
            raw_perfs[dname].values.flatten(), naucs.loc[dname].values.flatten()
        )[0]

    corr_df.loc["General"] = pearsonr(raw_perfs.values.flatten(), naucs.T.values.flatten())[0]

    return corr_df


def make_calibration_study(
    all_data,
    abstention_rates,
    dataset_name,
    model_name,
    metric,
    methods,
    random_seeds,
    test_size=0.2,
    alpha=0.1,
    quantile_bad=0.1,
    quantile_good=0.9,
    quantile_score=0,
):
    thold_calibration_study = {method: np.zeros((len(random_seeds), len(abstention_rates))) for method in methods}
    perf_calibration_study = {method: np.zeros((len(random_seeds), len(abstention_rates))) for method in methods}
    rel_scores, targets = all_data[model_name][dataset_name].values()
    metrics = evaluate_instances(rel_scores, targets, metric)

    # Evaluate threshold and abstention calibration for various random seeds
    for seed in random_seeds:
        rel_scores_ref, rel_scores_test, metrics_ref, metrics_test = train_test_split(
            rel_scores, metrics, test_size=test_size, random_state=seed
        )  # ref/test split

        for method in methods:
            conf_scorer = AbstentionReranker(
                method, metric, alpha, quantile_bad, quantile_good, quantile_score
            )  # initialize confidence scorer
            conf_scorer.get_scorer(rel_scores_ref, metrics_ref)  # fit scorer (when relevant)
            conf_scores_ref = conf_scorer.compute_confidence_scores(
                rel_scores_ref
            )  # compute confidence scores for ref set
            conf_scores_test = conf_scorer.compute_confidence_scores(
                rel_scores_test
            )  # compute confidence scores for test set
            abstention_thresholds = np.quantile(conf_scores_ref, abstention_rates)

            for i, (rate, threshold) in enumerate(zip(abstention_rates, abstention_thresholds)):
                thold_calibration_study[method][seed, i] = rate - len(
                    conf_scores_test[conf_scores_test < threshold]
                ) / len(conf_scores_test)
                perf_calibration_study[method][seed, i] = (
                    metrics_ref[conf_scores_ref >= threshold].mean()
                    - metrics_test[conf_scores_test >= threshold].mean()
                )

    # Summarize results in a summary data frame
    calib_df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [(study, method) for study in ["Abstention Rate", f"m{metric}"] for method in methods]
        ),
        columns=abstention_rates,
    )
    calib_df.columns.name = "Target Abstention Rate"
    calib_df.index.name = "Method"

    for method in methods:
        abs_diff_rate_mean = np.abs(thold_calibration_study[method]).mean(axis=0)
        abs_diff_perf_mean = np.abs(perf_calibration_study[method]).mean(axis=0)
        calib_df.loc[("Abstention Rate", method)] = abs_diff_rate_mean
        calib_df.loc[(f"m{metric}", method)] = abs_diff_perf_mean

    return calib_df


def make_ref_size_study(
    all_data,
    abstention_rates,
    dataset_names,
    model_name,
    metric,
    methods,
    random_seeds,
    test_size=0.2,
    alpha=0.1,
    quantile_bad=0.1,
    quantile_good=0.9,
    quantile_score=0
):
    ref_subsample_sizes = np.array([0.5**k for k in range(10)])
    ref_size_study = {
        dataset_name: {method: np.zeros((len(random_seeds), len(ref_subsample_sizes))) for method in methods}
        for dataset_name in dataset_names
    }
    ref_sizes = {dataset_name: np.zeros(len(ref_subsample_sizes)) for dataset_name in dataset_names}

    for j, size in enumerate(ref_subsample_sizes):
        for i, seed in enumerate(random_seeds):
            strat_evals = evaluate_strategies_on_benchmark(
                all_data,
                abstention_rates,
                dataset_names,
                [model_name],
                [metric],
                methods,
                [seed],
                test_size,
                alpha,
                quantile_bad,
                quantile_good,
                quantile_score,
                size,
            )
            naucs = compute_naucs(strat_evals, abstention_rates, dataset_names, [model_name], [metric], methods)

            for dataset_name in dataset_names:
                for method in methods:
                    ref_size_study[dataset_name][method][i, j] = naucs.loc[(dataset_name, metric), (model_name, method)]

    for dataset_name in dataset_names:
        ref_sizes[dataset_name] = np.maximum(
            np.round(all_data[model_name][dataset_name]["scores"].shape[0] * ref_subsample_sizes * (1 - test_size)), 1
        )

    return ref_size_study, ref_sizes


def get_minimum_reference_sizes(ref_size_study, ref_sizes, ref_free_method, ref_based_method):
    dataset_names = list(ref_sizes.keys())
    min_ref_sizes = pd.DataFrame(index=dataset_names, columns=["Break-Even"])
    min_ref_sizes.index.name = "Dataset"

    for dataset_name in dataset_names:
        min_ref_sizes.loc[dataset_name, "Break-Even"] = int(
            ref_sizes[dataset_name][
                ref_size_study[dataset_name][ref_based_method].mean(axis=0)
                > ref_size_study[dataset_name][ref_free_method].mean(axis=0)
            ][-1]
        )

    min_ref_sizes.loc["Average"] = min_ref_sizes.mean()

    return min_ref_sizes


def make_runtime_study(model_path, dataset_path, conf_scorer, num_trials=100, seed=0):
    model = SentenceTransformer(model_path)
    queries, positives, negatives = load_reranking_dataset(dataset_path)
    queries, positives, negatives = process_dataset(queries, positives, negatives)
    runtime_study = pd.DataFrame(0.0, index=range(num_trials), columns=["Relevance scores", "Confidence scores"])
    np.random.seed(seed)

    for i in range(num_trials):
        idx = np.random.randint(len(queries))
        q, pos, neg = queries[idx], positives[idx], negatives[idx]
        doc_emb = model.encode([doc for doc in (pos + neg)], normalize_embeddings=True)

        start = time.time()
        q_emb = model.encode(q).reshape(1, -1)
        rel_scores_instance = util.cos_sim(q_emb, doc_emb).numpy()
        runtime_study.loc[i, "Relevance scores"] = time.time() - start

        start = time.time()
        conf_scores_instance = conf_scorer.compute_confidence_scores(rel_scores_instance)
        runtime_study.loc[i, "Confidence scores"] = time.time() - start

    runtime_study_summary = pd.DataFrame(index=["Avg. runtime (ms)"], columns=runtime_study.columns)
    runtime_study_summary.iloc[0] = runtime_study.mean(axis=0) * 1000
    runtime_study_summary["Extra Time"] = (
        runtime_study_summary["Confidence scores"] / runtime_study_summary["Relevance scores"]
    )

    return runtime_study_summary


def make_domain_adaptation_study(
    all_data,
    abstention_rates,
    dataset_names,
    model_names,
    metric_names,
    methods,
    alpha=0.1,
    quantile_bad=0.1,
    quantile_good=0.9,
    quantile_score=0,
):
    dom_adap_study = []

    for dname_ref in dataset_names:
        dom_adap_study_dat = {
            model_name: {
                dname_test: {
                    metric: {method: None for method in methods + ["oracle", "random"]} for metric in metric_names
                }
                for dname_test in dataset_names
            }
            for model_name in model_names
        }
        dnames_test = [dname for dname in dataset_names if dname != dname_ref]

        for model_name in model_names:
            rel_scores_ref, targets_ref = all_data[model_name][dname_ref].values()

            for dname_test in dnames_test:
                rel_scores_test, targets_test = all_data[model_name][dname_test].values()

                for metric in metric_names:
                    metrics_ref = evaluate_instances(rel_scores_ref, targets_ref, metric)
                    metrics_test = evaluate_instances(rel_scores_test, targets_test, metric)
                    oracle_eval = evaluate_oracle(metrics_test, abstention_rates)
                    dom_adap_study_dat[model_name][dname_test][metric]["oracle"] = oracle_eval.reshape(1, -1)
                    dom_adap_study_dat[model_name][dname_test][metric]["random"] = np.array(
                        [[oracle_eval[0]] * len(abstention_rates)]
                    )

                    for method in methods:
                        conf_scorer = AbstentionReranker(method, metric, alpha, quantile_bad, quantile_good, quantile_score)
                        conf_scorer.get_scorer(rel_scores_ref, metrics_ref)
                        conf_scores_test = conf_scorer.compute_confidence_scores(rel_scores_test)
                        strat_eval = evaluate_strategy(conf_scores_test, metrics_test, abstention_rates)
                        dom_adap_study_dat[model_name][dname_test][metric][method] = strat_eval.reshape(1, -1)

        naucs_df = compute_naucs(dom_adap_study_dat, abstention_rates, dnames_test, model_names, metric_names, methods)
        naucs_df.index = pd.MultiIndex.from_tuples(
            [(dname_ref, dname_test, metric) for dname_test, metric in naucs_df.index],
            names=["Reference set", "Test set", "Metric"],
        )
        dom_adap_study.append(naucs_df)

    return pd.concat(dom_adap_study)


def make_instance_qualification_study(
    all_data,
    abstention_rates,
    dataset_names,
    model_names,
    metric_names,
    methods,
    quantiles,
    random_seeds,
    test_size=0.2,
    alpha=0.1,
    quantile_score=0,
):
    inst_qual_study = {
        model_name: {
            dataset_name: {
                metric: {method: np.zeros((len(random_seeds), len(quantiles))) for method in methods}
                for metric in metric_names
            }
            for dataset_name in dataset_names
        }
        for model_name in model_names
    }

    for j, quantile in enumerate(quantiles):
        quantile_bad, quantile_good = quantile, 1 - quantile

        for i, seed in enumerate(random_seeds):
            strat_evals_quantile = evaluate_strategies_on_benchmark(
                all_data,
                abstention_rates,
                dataset_names,
                model_names,
                metric_names,
                methods,
                [seed],
                test_size,
                alpha,
                quantile_bad,
                quantile_good,
                quantile_score,
                None,
            )
            naucs_quantile = compute_naucs(
                strat_evals_quantile, abstention_rates, dataset_names, model_names, metric_names, methods
            )

            for model_name in model_names:
                for dataset_name in dataset_names:
                    for metric in metric_names:
                        for method in methods:
                            inst_qual_study[model_name][dataset_name][metric][method][i, j] = naucs_quantile.loc[
                                (dataset_name, metric), (model_name, method)
                            ]

    return inst_qual_study


def make_linreg_weights_analysis(all_data, dataset_names, model_names, metric_names, alpha=0.1):
    weights_analysis = {
        model_name: {dataset_name: {metric: None for metric in metric_names} for dataset_name in dataset_names}
        for model_name in model_names
    }

    for model_name in model_names:
        for dataset_name in dataset_names:
            rel_scores, targets = all_data[model_name][dataset_name].values()

            for metric in metric_names:
                metrics = evaluate_instances(rel_scores, targets, metric)
                _, linreg_coefs = get_scorer_linreg(rel_scores, metrics, alpha=alpha, return_coefs=True)
                weights_analysis[model_name][dataset_name][metric] = linreg_coefs

    return weights_analysis
