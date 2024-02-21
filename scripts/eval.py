# Directory path
import os 
# add project root directory to the path
DIR_PATH = os.path.dirname(os.path.realpath('__file__'))

# Import relevant packages
import sys
sys.path.append(DIR_PATH)

import numpy as np

from abstention_reranker.utils import *
from abstention_reranker.abst_utils import *
from abstention_reranker.eval_utils import *
from abstention_reranker.plot_utils import *

import warnings
warnings.filterwarnings('ignore')

import argparse

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', type=str, default='baselines')
parser.add_argument('-qb', '--quantile_bad', type=float, default=0.1)
parser.add_argument('-qs', '--quantile_score', type=float, default=0.)
parser.add_argument('-t', '--temperature', type=float, default=1.)
args = parser.parse_args()
print(f'{args}\n')

# Model names and dataset names
dataset_names = ['SciDocs', 'AskUbuntu', 'StackOverflow', 'Alloprof', 'CMedQAv1', 'Mmarco']
model_names = ['ember-v1', 
    'llm-embedder',
    'bge-base-en-v1.5',
    'bge-reranker-base',
    'bge-reranker-large', 
    'e5-small-v2', 
    'e5-large-v2',
    'multilingual-e5-small', 
    'multilingual-e5-large',  
    'msmarco-MiniLM-L6-cos-v5', 
    'msmarco-distilbert-dot-v5',
    'ms-marco-TinyBERT-L-2-v2', 
    'ms-marco-MiniLM-L-6-v2', 
    'stsb-TinyBERT-L-4',
    'stsb-distilroberta-base', 
    'multi-qa-distilbert-cos-v1', 
    'multi-qa-MiniLM-L6-cos-v1',
    'all-MiniLM-L6-v2', 
    'all-distilroberta-v1',      
    'all-mpnet-base-v2', 
    'quora-distilroberta-base', 
    'qnli-distilroberta-base'
]

# Store relevance scores datasets in all_data dictionary
all_data = load_relevance_scores_datasets_from_local(model_names, dataset_names, path='../data')

if args.method == 'baselines':
    # Evaluate abstention strategies on benchmark
    strat_evals = evaluate_strategies_on_benchmark(
        all_data=all_data, 
        abstention_rates=np.linspace(0, 0.8, 9), 
        dataset_names=dataset_names,
        model_names=model_names,
        metric_names=['AP', 'NDCG', 'RR'],
        methods=['max', 'std', '1-2', 'linreg'],
        random_seeds=np.arange(5),
        quantile_bad=args.quantile_bad,
        quantile_good=1-args.quantile_bad,
        quantile_score=args.quantile_score,
    )

    # Compute nAUCs
    naucs = compute_naucs(
        strategy_evaluations=strat_evals, 
        abstention_rates=np.linspace(0, 0.8, 9),
        dataset_names=dataset_names,
        model_names=model_names,
        metric_names=['AP', 'NDCG', 'RR'],
        methods=['max', 'std', '1-2', 'linreg'],
    )

else:
    # Apply softmax to relevance scores
    for mname in model_names:
        for dname in dataset_names:
            rel_scores = all_data[mname][dname]['scores']
            all_data[mname][dname]['scores'] = softmax(rel_scores, args.temperature)

    # Evaluate abstention strategies on benchmark
    strat_evals = evaluate_strategies_on_benchmark(
        all_data=all_data, 
        abstention_rates=np.linspace(0, 0.8, 9), 
        dataset_names=dataset_names,
        model_names=model_names,
        #metric_names=['AP', 'NDCG', 'RR'],
        metric_names=['AP'],
        methods=[args.method],
        random_seeds=np.arange(5),
        quantile_bad=args.quantile_bad,
        quantile_good=1-args.quantile_bad,
        quantile_score=args.quantile_score,
    )

    # Compute nAUCs
    naucs = compute_naucs(
        strategy_evaluations=strat_evals, 
        abstention_rates=np.linspace(0, 0.8, 9),
        dataset_names=dataset_names,
        model_names=model_names,
        #metric_names=['AP', 'NDCG', 'RR'],
        metric_names=['AP'],
        methods=[args.method],
    )

# Save as .csv
naucs.to_csv('../evals/' + args.method + f'_qb{args.quantile_bad}_qs{args.quantile_score}_t{args.temperature}' * (args.method != 'baselines') + '.csv')