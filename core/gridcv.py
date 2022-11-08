from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, chi2
from sklearn.decomposition import PCA
from constants import GRID_CONFIG_MODELS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.impute import SimpleImputer


class GridSearchBase():

    def __init__(self, X, y, pca_level=15,random_state=42, n_splits =10, scaling=True, oversampling =None):
            self.X =X
            self.y =y
            self.random_state =random_state
            self.pca_level=pca_level
            self.grid_methods ={
                                "xgb": XGBClassifier(),
                                "svm": SVC(),
                                "rf" : RandomForestClassifier(),
                                "lr" : LogisticRegression()
                }
            self.params_grids =GRID_CONFIG_MODELS
            self.__generate_dimension_methods__()
            self.kfolds = StratifiedKFold(n_splits, shuffle=True, random_state=self.random_state)
            self.best_params =[]
            self.best_models =[]
            self.best_quality = []
            self.best_feature_selection =[]
            self.scaling =scaling
            self.oversampling =oversampling

    def __generate_dimension_methods__(self):
        self.fearure_selection_methods = [SelectKBest(f_classif, k='all')]
        if self.X.shape[1] > self.X.shape[0]:
            self.n_features = [int(self.X.shape[0]*0.8), self.X.shape[0]]
            self.fearure_selection_methods += [SelectKBest(score_func=f_classif, k=n) for n in self.n_features]
            self.fearure_selection_methods += [SelectFromModel(
                                                        estimator =RandomForestClassifier(n_estimators=int(self.X.shape[0] ** 0.5),random_state=self.random_state),
                                                        max_features=n) for n in self.n_features]
            self.fearure_selection_methods += [SelectFromModel(
                                                        estimator =LogisticRegression(random_state=self.random_state),
                                                        max_features=n) for n in self.n_features]
            self.fearure_selection_methods += [PCA(self.pca_level, random_state=self.random_state)]

    def train(self):
        for model_name, model in self.grid_methods.items():
            best_model, best_qualuty, best_params,best_feature_selection =None, 0,None,None
            for feature_selection_method in self.fearure_selection_methods:
                print(f"model_name {model_name}, feature_selection_method {feature_selection_method}")
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
                search = GridSearchCV(pipe, self.params_grids[model_name], cv=self.kfolds,
                                      scoring=['accuracy', 'roc_auc', 'f1_macro'],
                                      refit='f1_macro', return_train_score=True, verbose=0).fit(self.X, self.y)
                print("ROC AUC 10 folds: {} +- {} std".
                      format(search.cv_results_['mean_test_roc_auc'][search.best_index_],
                             search.cv_results_['std_test_roc_auc'][search.best_index_]))
                print("Accuracy 10 folds: {} +- {} std".
                      format(search.cv_results_['mean_test_accuracy'][search.best_index_],
                             search.cv_results_['std_test_accuracy'][search.best_index_]))
                print("F1 10 folds: {} +- {} std".
                      format(search.cv_results_['mean_test_f1_macro'][search.best_index_],
                             search.cv_results_['std_test_f1_macro'][search.best_index_]))
                if search.cv_results_['mean_test_f1_macro'][search.best_index_]>best_qualuty:
                    best_model=model
                    best_feature_selection =feature_selection_method
                    best_params=search.best_params_
                    best_qualuty =search.cv_results_['mean_test_f1_macro'][search.best_index_]
                print("__________________________________________________________________________")

            self.best_quality.append(best_qualuty)
            self.best_params.append(best_params)
            self.best_models.append(best_model)
            self.best_feature_selection.append(best_feature_selection)

        return self.sorted_results()

    def sorted_results(self):
        results_list =zip(self.best_models,self.best_params, self.best_quality,self.best_feature_selection)
        results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
        return results_list

