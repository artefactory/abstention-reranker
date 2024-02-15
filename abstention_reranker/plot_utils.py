import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

colors_dict = {
    "max": "orange",
    "std": "red",
    "1-2": "green",
    "linreg": "blue",
    "logreg": "purple",
    "mahalanobis": "black",
}
markers_dict = {"max": "o", "std": "o", "1-2": "o", "linreg": "d", "ridge": "d", "logreg": "d", "mahalanobis": "d"}


def plot_performance_vs_abstention(
    strategy_evaluations, abstention_rates, dataset_name, model_name, metric_names, methods, savefig=False, path=None
):
    strat_eval_mod_dat = strategy_evaluations[model_name][dataset_name]
    fontsize = 14
    sns.set(style="white", palette="bright")
    plt.figure(figsize=(12, 3))

    for i, metric in enumerate(metric_names):
        plt.subplot(1, 3, i + 1)
        plt.plot(
            abstention_rates,
            strat_eval_mod_dat[metric]["oracle"].mean(axis=0),
            linestyle="dashed",
            c="gray",
            label=r"$u^{*}$",
        )
        plt.plot(
            abstention_rates,
            strat_eval_mod_dat[metric]["random"].mean(axis=0),
            linestyle="dotted",
            c="gray",
            label=r"$\tilde{u}$",
        )

        for method in methods:
            plt.plot(
                abstention_rates,
                strat_eval_mod_dat[metric][method].mean(axis=0),
                c=colors_dict[method],
                marker=markers_dict[method],
                markersize=5,
                label="$u_{" + method[:3] + "}$",
            )

        plt.xlabel("Abstention Rate", fontsize=fontsize)
        plt.ylabel("m" + metric, fontsize=fontsize)

    plt.legend(ncols=7, bbox_to_anchor=(0.65, -0.25), fontsize=fontsize)
    plt.subplots_adjust(wspace=0.4)
    sns.despine()
    plt.tight_layout()

    if savefig:
        plt.savefig(path, bbox_inches="tight")


def plot_abstention_vs_raw_performance(raw_perfs, naucs, metric, method, savefig=False, path=None):
    model_names = list(raw_perfs.index)
    dataset_names = raw_perfs.columns.get_level_values(0).unique()

    # Create comparison data frame
    df = pd.DataFrame(columns=["model", "dataset", "no_abst_perf", "nauc"])
    counter = 0

    for mname in model_names:
        for dname in dataset_names:
            df.loc[counter, "model"] = mname
            df.loc[counter, "dataset"] = dname
            df.loc[counter, "no_abst_perf"] = raw_perfs.loc[mname, (dname, metric)]
            df.loc[counter, "nauc"] = naucs.loc[(dname, metric), (mname, method)]
            counter += 1

    # Generate plot
    fontsize = 16
    markers = ["o", "^", "s", "p", "d", "X"]

    sns.set(style="white", palette="bright")
    plt.figure(figsize=(5, 5))

    for i, dname in enumerate(raw_perfs.columns.get_level_values(0).unique()):
        df_dat = df[df.dataset == dname]
        plt.scatter(df_dat.no_abst_perf, df_dat.nauc, marker=markers[i], label=dname)

    plt.xlabel(f"no-abstention m{metric}", fontsize=fontsize)
    plt.ylabel("nAUC", fontsize=fontsize)
    plt.legend(ncols=2, fontsize=fontsize, bbox_to_anchor=(1.1, -0.2))
    sns.despine()
    plt.tight_layout()

    if savefig:
        plt.savefig(path)


def plot_naucs_vs_ref_size(ref_size_study, ref_sizes, dataset_names, methods, savefig=False, path=None):
    fontsize = 18
    sns.set(style="white", palette="bright")
    plt.figure(figsize=(7, 10))

    for i, dataset_name in enumerate(dataset_names):
        plt.subplot(3, 2, i + 1)

        for method in methods:
            naucs_mean = ref_size_study[dataset_name][method].mean(axis=0)
            naucs_std = ref_size_study[dataset_name][method].std(axis=0)
            plt.plot(
                ref_sizes[dataset_name],
                naucs_mean,
                color=colors_dict[method],
                marker=markers_dict[method],
                markersize=5,
                label="$u_{" + method[:3] + "}$",
            )
            plt.fill_between(
                ref_sizes[dataset_name],
                naucs_mean - naucs_std,
                naucs_mean + naucs_std,
                color=colors_dict[method],
                alpha=0.3,
            )

        if i // 2 == 2:
            plt.xlabel("Reference Size", fontsize=fontsize)
        if i % 2 == 0:
            plt.ylabel("nAUC", fontsize=fontsize)

        plt.title(dataset_name, fontsize=fontsize)

    plt.legend(loc="lower right", fontsize=fontsize)
    sns.despine()
    plt.tight_layout()

    if savefig:
        plt.savefig(path)


def plot_naucs_vs_qualification_threshold(
    inst_qual_study, quantiles, model_name, dataset_names, metric, methods, savefig=False, path=None
):
    fontsize = 16
    sns.set(style="white", palette="bright")
    plt.figure(figsize=(7, 10))

    for i, dataset_name in enumerate(dataset_names):
        plt.subplot(3, 2, i + 1)

        for method in methods:
            naucs_mean = inst_qual_study[model_name][dataset_name][metric][method].mean(axis=0)
            naucs_std = inst_qual_study[model_name][dataset_name][metric][method].std(axis=0)
            plt.plot(
                quantiles,
                naucs_mean,
                color=colors_dict[method],
                marker=markers_dict[method],
                markersize=5,
                label="$u_{" + method[:3] + "}$",
            )
            plt.fill_between(
                quantiles, naucs_mean - naucs_std, naucs_mean + naucs_std, color=colors_dict[method], alpha=0.3
            )

        if i // 2 == 2:
            plt.xlabel("Qualification Threshold", fontsize=fontsize)
        if i % 2 == 0:
            plt.ylabel("nAUC", fontsize=fontsize)

        plt.title(dataset_name, fontsize=fontsize)

    plt.legend(loc="lower right", fontsize=fontsize)
    sns.despine()
    plt.tight_layout()

    if savefig:
        plt.savefig(path)


def plot_linreg_coefficients(
        linreg_weights_analysis, model_name, dataset_names, metric, savefig=False, path=None
):
    fontsize = 16
    sns.set(style="white", palette="bright")
    plt.figure(figsize=(7, 10))

    for i, dataset_name in enumerate(dataset_names):    
        coefs = linreg_weights_analysis[model_name][dataset_name]
        
        plt.subplot(3,2,i+1)
        plt.bar(range(len(coefs[metric])), coefs[metric])

        if i // 2 == 2:
            plt.xlabel("Document", fontsize=fontsize)
        if i % 2 == 0:
            plt.ylabel("Coefficient Value", fontsize=fontsize)

        sns.despine()
        plt.tight_layout()

        if savefig:
            plt.savefig(path)


def plot_runtime_comparison(runtime_study, savefig=False, path=None):
    fontsize = 20
    plt.rcParams["figure.figsize"] = [8, 2.2]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams['text.usetex'] = False
    plt.rcParams['axes.facecolor'] = 'none'
    plt.rcParams.update({'font.size': 16})

    rel_sores_time = runtime_study.iloc[0,0]
    conf_scores_time = runtime_study.iloc[0,1]
    
    b1 = plt.barh([""], [rel_sores_time], color="gray", alpha=0.5)
    b2 = plt.barh([""], [conf_scores_time], left=[rel_sores_time], color="blue", alpha=0.5)

    plt.legend([b1, b2], [r"$g_{\bf{\theta}}$", r"$u_{lin}$"], title="", fontsize=fontsize, ncol=2, bbox_to_anchor=(0.27, 1))
    plt.xlabel("Avg. Runtime (ms)", fontsize=fontsize)
    plt.xlim(-0.025, rel_sores_time + conf_scores_time + 0.025)
    
    if savefig:
        plt.savefig(path, bbox_inches='tight')
    
    plt.show()
