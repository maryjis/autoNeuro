import time
import datetime

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, chi2
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.impute import SimpleImputer

from .constants import GRID_CONFIG_MODELS


class GridSearchBase:
    def __init__(
        self,
        X,
        y,
        pca_level=15,
        random_state=42,
        n_splits=10,
        scaling=True,
        oversampling=None,
    ):
        self.X = X
        self.y = y
        self.random_state = random_state
        self.pca_level = pca_level
        self.scaling = scaling
        self.oversampling = oversampling

        # model types
        self.grid_methods = {
            "xgb": XGBClassifier(),
            "svm": SVC(),
            "rf" : RandomForestClassifier(),
            "lr" : LogisticRegression(),
        }
        
        # hardcoded configs 
        self.params_grids = GRID_CONFIG_MODELS

        self.__generate_dimension_methods__()
        
        # k-folder
        self.kfolds = StratifiedKFold(n_splits, shuffle=True, random_state=self.random_state)
        
        # final results
        self.best_params = []
        self.best_models = []
        self.best_quality = []
        self.best_feature_selection = []
        self.total_metrics = {}

    def __generate_dimension_methods__(self):
        self.feature_selection_methods = [SelectKBest(f_classif, k='all')]
        
        # num features > num samples
        if self.X.shape[1] > self.X.shape[0]:
            self.n_features = [int(self.X.shape[0] * 0.8), self.X.shape[0]]
            
            # create a list of feature selection algos 
            self.feature_selection_methods += [SelectKBest(score_func=f_classif, k=n) for n in self.n_features]
            self.feature_selection_methods += [
                SelectFromModel(
                    estimator=RandomForestClassifier(n_estimators=int(self.X.shape[0] ** 0.5), random_state=self.random_state),
                    max_features=n,
                )
                for n in self.n_features
            ]
            self.feature_selection_methods += [
                SelectFromModel(
                    estimator=LogisticRegression(random_state=self.random_state),
                    max_features=n,
                )
                for n in self.n_features
            ]
            #self.feature_selection_methods += [PCA(self.pca_level, random_state=self.random_state)]

    def train(self):
        # for each model for each feature selection method
        for model_name, model in self.grid_methods.items():
            best_model, best_quality, best_params,best_feature_selection = None, 0, None, None
            for feature_selection_method in self.feature_selection_methods:
                start = time.time()
                pipe_desc = f"model_name: {model_name}, feature_selection_method: {feature_selection_method}" 
                print(pipe_desc)
                
                # create a sklearn pipeline
                if self.scaling:
                    pipe = Pipeline([
                                       ('scaler', StandardScaler()),
                                       ("feature_selection", feature_selection_method),
                                       ('model', model)])
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
                print(pipe)

                # run sklearn grid search w/ a given pipeline
                metric_names = ['accuracy', 'roc_auc', 'f1_macro']
                search = GridSearchCV(
                    pipe,
                    self.params_grids[model_name],
                    cv=self.kfolds,
                    scoring=metric_names,
                    refit='f1_macro',
                    return_train_score=True,
                    verbose=0,
                ).fit(self.X, self.y)
                
                # print mean cv metrics with std
                print("ROC AUC 10 folds: {} +- {} std".
                      format(search.cv_results_['mean_test_roc_auc'][search.best_index_],
                             search.cv_results_['std_test_roc_auc'][search.best_index_]))
                print("Accuracy 10 folds: {} +- {} std".
                      format(search.cv_results_['mean_test_accuracy'][search.best_index_],
                             search.cv_results_['std_test_accuracy'][search.best_index_]))
                print("F1 10 folds: {} +- {} std".
                      format(search.cv_results_['mean_test_f1_macro'][search.best_index_],
                             search.cv_results_['std_test_f1_macro'][search.best_index_]))

                # store test metrics for that pipeline
                metrics = {}
                for stat in ['mean', 'std']:
                    for metric_name in metric_names:
                        test_metric_name =  f'{stat}_test_{metric_name}'
                        metrics[test_metric_name] = search.cv_results_[test_metric_name][search.best_index_]
                self.total_metrics[pipe_desc] = metrics

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

