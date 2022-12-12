import time
import datetime
from typing import Dict, List
from tqdm import tqdm

from frozendict import frozendict
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, chi2, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap

from .constants import GRID_CONFIG_MODELS

DEFAULT_MODELS = {
    'lgb': LGBMClassifier(),
    "xgb": XGBClassifier(),
    "svm": SVC(),
    "rf" : RandomForestClassifier(),
    "lr" : LogisticRegression(),
}


def report_time(f):
    def inner(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        elapsed_time = datetime.timedelta(seconds=round(end - start))
        print(f'Time elapsed: {elapsed_time}')
        return res
    return inner


def get_metrics_from_search(
    search: GridSearchCV,
    metric_names: List[str],
    split_name: str = 'test',
) -> Dict[str, float]:
    ''' Extract metric values for the best model from GridSearchCV object '''
    metrics = {}
    for stat in ['mean', 'std']:
        for metric_name in metric_names:
            test_metric_name =  f'{stat}_{split_name}_{metric_name}'
            metrics[test_metric_name] = search.cv_results_[test_metric_name][search.best_index_]
    return metrics


def print_metrics(metrics_dict, metric_names, split_name: str = 'test'):
    ''' `metrics_dict` is a result of get_metrics_from_search '''
    print('Best metrics on test')
    for metric_name in metric_names:
        mean = metrics_dict[f'mean_{split_name}_{metric_name}']
        std = metrics_dict[f'std_{split_name}_{metric_name}']
        print(f'\t{metric_name}: {mean:.2f} ± {std:.2f}')


def compute_fold_metrics(pipe, X, y):
    labels_pred = pipe.predict(X)
    proba_pred = pipe.predict_proba(X)[:, 1]

    return {
        'f1': f1_score(y, labels_pred),
        'roc_auc': roc_auc_score(y, proba_pred),
        'acc': accuracy_score(y, labels_pred),
    }


def create_pipeline(
    model,
    feature_selection_method,
    scaling=False,
    oversampling=False,
):
    if isinstance(model, str):
        model = DEFAULT_MODELS[model]

    if scaling:
        pipe = Pipeline([
           ('scaler', StandardScaler()),
           ("feature_selection", feature_selection_method),
           ('model', model)]
        )
        if oversampling:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('oversampling', oversampling),
                ("feature_selection", feature_selection_method),
                ('model', model)])
    else:
        pipe = Pipeline([
            ("feature_selection", feature_selection_method),
            ('model', model)])

        if oversampling:
            pipe = Pipeline([
                ('oversampling', oversampling),
                ("feature_selection", feature_selection_method),
                ('model', model)])
    return pipe


class GridSearchBase:
    default_models = DEFAULT_MODELS

    def __init__(
        self,
        X,
        y,
        model_names=None,
        params_grids=None,
        pca_level=15,
        random_state=42,
        n_splits=10,
        internal_n_splits=5,
        scaling=True,
        oversampling=None,
        n_jobs=1,
    ):
        self.X = X
        self.y = y
        self.random_state = random_state
        self.pca_level = pca_level
        self.scaling = scaling
        self.oversampling = oversampling
        self.n_jobs = n_jobs

        # model types
        if model_names is None:
            self.models = self.default_models
        else:
            self.models = {name: self.default_models[name] for name in model_names}

        # hardcoded configs
        self.params_grids  = params_grids
        if self.params_grids is None:
            self.params_grids = GRID_CONFIG_MODELS

        self.create_feature_selection_methods()

        # kfolds
        self.kfolds = StratifiedKFold(n_splits, shuffle=True, random_state=self.random_state)
        self.internal_kfolds = StratifiedKFold(internal_n_splits, shuffle=True, random_state=self.random_state)
        self.metric_names = ['accuracy', 'roc_auc', 'f1_macro']

        # final results
        self.best_params = []
        self.best_models = []
        self.best_quality = []
        self.best_feature_selection = []
        self.total_metrics = {}

    def select_features(self):
        Xs = []

    def create_feature_selection_methods(self, feature_reductions =True):
        #self.feature_selection_methods = [SelectKBest(f_classif, k='all')]
        self.feature_selection_methods = [VarianceThreshold(threshold=0.0)]

        # num features > num samples
        if self.X.shape[1] > self.X.shape[0]:
            self.n_features = [round(self.X.shape[0] * 0.3), round(self.X.shape[0] * 0.1), 3]

            if feature_reductions:
                self.feature_selection_methods +=[Isomap(n_components=n) for n in self.n_features]
                self.feature_selection_methods +=[LocallyLinearEmbedding(n_components=n,method='modified', n_neighbors=n+5)
                                              for n in self.n_features]
                self.feature_selection_methods += [PCA(n, random_state=self.random_state) for n in self.n_features]
            else:
                self.feature_selection_methods += [
                    SelectFromModel(
                        estimator=LogisticRegression(random_state=self.random_state),
                        max_features=n,
                    )
                    for n in self.n_features
                ]
                self.feature_selection_methods += [SelectKBest(score_func=f_classif, k=n) for n in self.n_features]
                self.feature_selection_methods += [
                SelectFromModel(
                    estimator=RandomForestClassifier(n_estimators=int(self.X.shape[0] ** 0.5),
                                                     random_state=self.random_state,
                                                     ),
                    max_features=n,
                )
                for n in self.n_features
            ]

    def train(self):
        # for each model for each feature selection method
        for model_name, model in self.models.items():
            best_model, best_quality, best_params, best_feature_selection = None, 0, None, None
            for feature_selection_method in self.feature_selection_methods:
                start = time.time()
                pipe_desc = f'model_name: {model_name}, feature_selection_method: {feature_selection_method}'
                print(pipe_desc)

                pipe = self.create_pipeline(model, feature_selection_method)
                print(pipe)

                # run sklearn grid search w/ a given pipeline
                search = self.run_cv(self.X, self.y, pipe, model_name, self.kfolds)

                # get mean+std for metrics on test, save and print
                metrics = get_metrics_from_search(search, self.metric_names, split_name='test')
                self.total_metrics[pipe_desc] = metrics
                print_metrics(metrics, self.metric_names, split_name='test')

                # update best metrics based on f1 macro
                if search.cv_results_['mean_test_f1_macro'][search.best_index_] > best_quality:
                    best_model = model
                    best_feature_selection = feature_selection_method
                    best_params = search.best_params_
                    best_quality = search.cv_results_['mean_test_f1_macro'][search.best_index_]

                end = time.time()
                elapsed_time = datetime.timedelta(seconds=round(end-start))
                print(f'model: {pipe_desc}, time elapsed: {elapsed_time}')
                print("__________________________________________________________________________")

            self.best_quality.append(best_quality)
            self.best_params.append(best_params)
            self.best_models.append(best_model)
            self.best_feature_selection.append(best_feature_selection)

        return self.sorted_results(), self.total_metrics

    def sorted_results(self):
        # sort results by quality (f1)
        results_list = zip(self.best_models, self.best_params, self.best_quality, self.best_feature_selection)
        results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
        return results_list

    def nested_train(self):
        total_results = []
        for (model_name, pipe, pipe_params) in self.pipe_combinations():
            print(f'External CV for {pipe}')
            start = time.time()

            fold_metrics, best_params = self.nested_cv(pipe, model_name)
            fold_metrics = self.agg_fold_metrics(fold_metrics)
            res = {
                'model_name': model_name,
                'feat_select': pipe['feature_selection'],
                'best_params_per_fold': best_params,
            }
            res.update(fold_metrics)
            total_results.append(res)

            end = time.time()
            elapsed_time = datetime.timedelta(seconds=round(end - start))
            print(f'Pipe: {pipe}, elapsed time: {elapsed_time}')

        return sorted(total_results, key=lambda d: d['f1_mean'], reverse=True)

    def nested_cv(self, pipe, model_name):
        external_metrics = []
        best_params_list = []
        for train_idx, test_idx in self.kfolds.split(self.X, self.y):
            X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]
            X_test, y_test = self.X.iloc[test_idx], self.y.iloc[test_idx]

            # run gridsearch on train, get the best pipeline
            search = self.run_cv(X_train, y_train, pipe, model_name, self.internal_kfolds)
            best_pipe = search.best_estimator_
            best_params = search.best_params_

            # assess quality on test
            test_metrics = compute_fold_metrics(best_pipe, X_test, y_test)

            # update results
            external_metrics.append(test_metrics)
            best_params_list.append(frozendict(best_params))

        return external_metrics, best_params_list

    def agg_fold_metrics(self, metrics):
        stat_names = ['mean', 'std']
        df = pd.DataFrame(metrics)
        stats = df.agg(stat_names, axis=0)

        return {
            f'{c}_{s}': stats.loc[s, c]
            for c in df.columns
            for s in stat_names
        }

    def run_cv(self, X, y, pipe, model_name, kfolds):
        return GridSearchCV(
            pipe,
            self.params_grids[model_name],
            cv=kfolds,
            scoring=self.metric_names,
            refit='f1_macro',
            return_train_score=True,
            n_jobs=self.n_jobs,
            verbose=0,
        ).fit(X, y)

    def pipe_combinations(self):
        for model_name, model in self.models.items():
            for feature_selection_method in self.feature_selection_methods:
                pipe_params = self.params_grids[model_name]
                pipe = self.create_pipeline(model, feature_selection_method)
                yield (model_name, pipe, pipe_params)

    def create_pipeline(self, model, feature_selection_method):
        if self.scaling:
            pipe = Pipeline([
               ('scaler', StandardScaler()),
               ("feature_selection", feature_selection_method),
               ('model', model)]
            )
            if self.oversampling:
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('oversampling', self.oversampling),
                    ("feature_selection", feature_selection_method),
                    ('model', model)])
        else:
            pipe = Pipeline([
                ("feature_selection", feature_selection_method),
                ('model', model)])

            if self.oversampling:
                pipe = Pipeline([
                    ('oversampling', self.oversampling),
                    ("feature_selection", feature_selection_method),
                    ('model', model)])
        return pipe

    @property
    def supported_models(self):
        return list(self.default_models.keys())
