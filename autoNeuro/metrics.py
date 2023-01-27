from typing import Optional
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, classification_report, accuracy_score, roc_auc_score
from scikitplot.metrics import plot_roc_curve
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold


def get_feature_importances(pipe, X, max_features=None):
    if hasattr(pipe['feature_selection'], 'get_support'):
        col_idx = pipe['feature_selection'].get_support(indices=True)
        selected_features = list(X.columns[col_idx])

    # get feature importances from the model in the pipeline
    # as a list of tuples (name, importance)
    model = pipe['model']
    importances = []
    if hasattr(model, 'feature_importances_'):
        importances = list(zip(selected_features , model.feature_importances_))
        importances = sorted(importances, key=lambda x: abs(x[1]), reverse=True)[:max_features]
    elif hasattr(model, 'coef_'):
        importances = zip(selected_features, model.coef_[0, :])
        importances = sorted(importances, key=lambda x: abs(x[1]), reverse=True)[:max_features]

    return Counter([x[0] for x in importances])


class ExperimentsInfo:
    def __init__(self, X, y, pipe, experiment_name, random_state=42):
        self.X = X
        self.y = y
        self.pipe = pipe
        self.experiment_name = experiment_name
        self.counter = Counter()
        self.random_state = random_state

    def calculate_most_common_entitis(self, topN=20, show_roc_auc=False, result_path: Optional[Path] = None):
        # for each fold we save:
        tprs = []
        aucs = []
        predicted_values = []
        true_values = []
        prob_values = []
        confusions = []

        mean_fpr = np.linspace(0, 1, 100)

        # resplit data into folds
        kf = StratifiedKFold(n_splits=10, shuffle=True)

        if show_roc_auc:
            fig, ax = plt.subplots(figsize=(11, 10))

        # loop over folds
        for i, (train_index, test_index) in enumerate(kf.split(self.X, self.y)):
            # resplit into train and test
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            # refit to the current fold
            self.pipe = self.pipe.fit(X_train, y_train)

            # get the indices of the selected features and select them in X
            if hasattr(self.pipe['feature_selection'], 'get_support'):
                col_idx = self.pipe['feature_selection'].get_support(indices=True)
                feature_names_new = list(self.X.columns[col_idx])
            # PCA and the like (they don't select features, but create new ones)
            else:
                feature_names_new = [f'f_{i}' for i in range(self.pipe['feature_selection'].n_components_)]

            if show_roc_auc:
                viz = plot_roc_curve(self.pipe, X_test, y_test,
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3, lw=1, ax=ax, color='dimgray')

                # build an interpolated roc curve array from a roc curve plot
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            # predict labels and probas
            predict_values = self.pipe.predict(X_test)
            prob_value = self.pipe.predict_proba(X_test)

            predicted_values.append(predict_values)
            true_values.append(y_test)
            prob_values.append(prob_value)

            # get feature importances from the model in the pipeline
            # as a list of tuples (name, importance)
            importances = []
            if hasattr(self.pipe['model'], 'feature_importances_'):
                importances = list(zip(feature_names_new , self.pipe['model'].feature_importances_))
                importances = sorted(importances, key=lambda x: abs(x[1]), reverse=True)[:topN]
            elif hasattr(self.pipe['model'], 'coef_'):
                importances = zip(feature_names_new, self.pipe['model'].coef_[0, :])
                importances = sorted(importances, key=lambda x: abs(x[1]), reverse=True)[:topN]

            cm = confusion_matrix(y_test, predict_values, labels=self.pipe["model"].classes_)
            confusions.append(cm)

            for importance in importances:
                self.counter[importance[0]] += 1

        if show_roc_auc:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey',
                    label='Chance', alpha=.8)


            ax.plot(mean_fpr, mean_tpr, color='black',
                    label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                    lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                            label=r'$\pm$ 1 std. dev.')

            ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                   title="Receiver operating characteristic example")
            ax.legend(loc="lower right")

            # save roc_curve in a file
            if result_path:
                plt.savefig(result_path / 'roc_curve.pdf')
            plt.show()

        # sum all confusion matrices
        matr = confusions[0]
        for conf in confusions[1:]:
            matr += conf

        # concat predicts for all folds
        true_values = np.concatenate(true_values)
        predicted_values = np.concatenate(predicted_values)
        prob_values = np.concatenate(prob_values)

        # create a total report
        metric_dicts = classification_report(true_values, predicted_values, output_dict=True, zero_division=0)

        accuracy = accuracy_score(true_values, predicted_values)
        mean_auc = roc_auc_score(true_values, prob_values[:, 1])

        return metric_dicts, mean_auc, matr, accuracy

    def get_important_features(self, experiments_count: int, result_path: Optional[Path]=None):
        A_precisions = []
        A_recalls = []
        K_precisions = []
        K_recalls  = []
        mean_aucs = []
        confusions_all = []
        f1s = []
        accuracies = []

        # run `self.calculate_most_common_entitis` for each fold
        for i in range(experiments_count):
            # show roc auc for the last experiment
            if i == (experiments_count-1):
                metrics, mean_auc, confusions, accuracy = self.calculate_most_common_entitis(show_roc_auc=True, result_path=result_path)
            else:
                metrics, mean_auc, confusions, accuracy = self.calculate_most_common_entitis()

            confusions_all.append(confusions)
            A_precisions.append(metrics["1"]["precision"])
            A_recalls.append(metrics["1"]["recall"])
            K_precisions.append(metrics["0"]["precision"])
            K_recalls.append(metrics["0"]["recall"])
            f1s.append(metrics['macro avg']['f1-score'])
            mean_aucs.append(mean_auc)
            accuracies.append(accuracy)

        print("--------------------")

        # convert to numpy arrays
        A_precisions, A_recalls, K_precisions, K_recalls, mean_aucs, f1s,accuracies = np.array(A_precisions), np.array(
            A_recalls), np.array(K_precisions), np.array(K_recalls), np.array(mean_aucs), np.array(f1s), np.array(accuracies)

        results = self.generate_text_result(A_precisions, A_recalls, K_precisions, K_recalls, mean_aucs, accuracies, f1s)

        # count the most frequent features
        important_features = self.counter.most_common()[:20]
        important_features_df = pd.DataFrame.from_records(important_features, columns=["feature_name", "count"])

        return important_features_df, results

    def generate_text_result(self, A_precisions, A_recalls, K_precisions, K_recalls, mean_aucs, accuracies, f1s):
        text1 = f"{self.experiment_name}\n"
        text1 +=f"Pipeline: {self.pipe}\n"
        text1 +="____________________________________"
        text1 +="A precision mean {} +- {} std\n".format(A_precisions.mean(), A_precisions.std())
        text1 +="A recall mean {} +- {} std\n".format(A_recalls.mean(), A_recalls.std())
        text1 +="K precision mean {} +- {} std\n".format(K_precisions.mean(), K_precisions.std())
        text1 +="K recall mean {} +- {} std\n".format(K_recalls.mean(), K_recalls.std())
        text1 +="ROC AUC mean {} +- {} std\n".format(mean_aucs.mean(), mean_aucs.std())
        text1 +="f1-macro mean {} +- {} std\n".format(f1s.mean(), f1s.std())
        text1 +="Accuracy mean {} +- {} std\n".format(np.array(accuracies).mean(), np.array(accuracies).std())
        text1 += "__________________________________________"
        return text1
