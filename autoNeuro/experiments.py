import os
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .gridcv import GridSearchBase
from .metrics import ExperimentsInfo
from .stats import FeaturesStats


def warn(*args, **kwargs):
    pass

import warnings

warnings.warn = warn

EXPERIMENTS_PATH = os.environ.get('EXPERIMENTS_PATH', 'results')

if not os.path.exists(EXPERIMENTS_PATH):
    os.makedirs(EXPERIMENTS_PATH)


def combine_dataset(X, y, targets_name):
    dataset = X.copy()
    dataset['target'] = y.map({targets_name[0]: 'Control', targets_name[1]: 'Patient'})
    return dataset


def run(
    X,
    y,
    experiment_name,
    topN=3,
    repeats=10,
    scaling=True,
    targets_name=[0, 1],
    plot_density=True,
):
    dataset = combine_dataset(X, y, targets_name)

    grid_search = GridSearchBase(X, y, scaling=scaling)
    sorted_results = grid_search.train()

    important_features = {}

    result_path = Path(EXPERIMENTS_PATH) / experiment_name
    if not result_path.exists():
        os.mkdir(result_path)
    
    # select topN models from the grid search result
    best_f1, best_result = 0, None
    for i, result in enumerate(sorted_results[:topN]):
        model, model_params, qualuty, feature_selection = result
        if scaling:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ("feature_selection", feature_selection),
                ('model', model)])
        else:
            pipe = Pipeline([
                ("feature_selection", feature_selection),
                ('model', model)]
            )
 
        # recreate a pipeline
        pipe = pipe.set_params(**model_params)
 
        # given data and pipeline, compute metrics over folds and feature importances
        info = ExperimentsInfo(X, y, pipe, experiment_name=experiment_name)
        important_features_df, results = info.get_important_features(repeats)
        
        # save results for the best model
        if i == 0:
            best_f1 = qualuty
            best_result = results
        
        # compute some stats on data
        fs = FeaturesStats(dataset, important_features_df)
        important_features_df = fs.get_stats(plot_density=plot_density)
        important_features[str(model)] = important_features_df
        
        # save metrics report
        with open(result_path / f"model_best_{i}_metrics.txt", 'w') as fp:
            fp.write(results)
        
        # save feature importances
        important_features_df.to_excel(result_path /f"model_best_{i}_important_features.xls",  index=False)

    return best_result, best_f1
