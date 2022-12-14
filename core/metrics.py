from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import auc,plot_roc_curve,confusion_matrix,classification_report, accuracy_score, roc_auc_score
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold

class ExperimentsInfo():

    def __init__(self, X, y, pipe, experiment_name,random_state=42):
        self.X=X
        self.y=y
        self.pipe=pipe
        self.experiment_name =experiment_name
        self.counter = Counter()
        self.random_state =random_state

    def calculate_most_common_entitis(self, topN=20,show_roc_auc=False):
        tprs,aucs,predicted_values,true_values,prob_values,confusions = [],[],[],[],[], []
        mean_fpr = np.linspace(0, 1, 100)
        kf = StratifiedKFold(n_splits=10, shuffle=True,random_state=self.random_state)

        if show_roc_auc:
            fig, ax = plt.subplots(figsize=(11, 10))

        for i, (train_index, test_index) in enumerate(kf.split(self.X, self.y)):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]

            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            self.pipe = self.pipe.fit(X_train, y_train)

            cols = self.pipe['feature_selection'].get_support(indices=True)
            features_df_new = self.X.iloc[:,cols]

            if show_roc_auc:
                viz = plot_roc_curve(self.pipe, X_test, y_test,
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3, lw=1, ax=ax, color='dimgray')
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            predict_values = self.pipe.predict(X_test)
            prob_value =self.pipe.predict_proba(X_test)
            predicted_values.append(predict_values)
            true_values.append(y_test)
            prob_values.append(prob_value)

            importances =[]
            if hasattr(self.pipe['model'], 'feature_importances_'):
                importances = list(zip(list(features_df_new.columns), self.pipe['model'].feature_importances_))
                importances = sorted(importances, key=lambda x: abs(x[1]), reverse=True)[:topN]
            elif hasattr(self.pipe['model'], 'coef_'):
                importances =zip(list(features_df_new.columns),self.pipe['model'].coef_[0,:])
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
            plt.show()

        matr = confusions[0]
        for conf in confusions[1:]:
            matr += conf
        true_values =np.concatenate(true_values)
        predicted_values = np.concatenate(predicted_values)
        prob_values =np.concatenate(prob_values)
        metric_dicts =classification_report(true_values, predicted_values, output_dict=True, zero_division=0)

        accuracy =accuracy_score(true_values, predicted_values)
        mean_auc =roc_auc_score(true_values, prob_values[:,1])
        return metric_dicts, mean_auc, matr,accuracy


    def get_important_features(self,experiments_count):
        A_precisions, A_recalls, K_precisions, K_recalls, mean_aucs, confusions_all, f1s, accuracies = [], [], [], [], [], [], [], []
        for i in range(experiments_count):
            if i==(experiments_count-1):
                metrics, mean_auc, confusions,accuracy = self.calculate_most_common_entitis(show_roc_auc=True)
            else:
                metrics, mean_auc, confusions,accuracy = self.calculate_most_common_entitis()
            confusions_all.append(confusions)
            A_precisions.append(metrics["1"]["precision"])
            A_recalls.append(metrics["1"]["recall"])
            K_precisions.append(metrics["0"]["precision"])
            K_recalls.append(metrics["0"]["recall"])
            f1s.append(metrics['macro avg']['f1-score'])
            mean_aucs.append(mean_auc)
            accuracies.append(accuracy)
        print("--------------------")

        A_precisions, A_recalls, K_precisions, K_recalls, mean_aucs, f1s,accuracies = np.array(A_precisions), np.array(
            A_recalls), np.array(K_precisions), np.array(K_recalls), np.array(mean_aucs), np.array(f1s),np.array(accuracies)

        results =self.generate_text_result(A_precisions, A_recalls, K_precisions, K_recalls, mean_aucs, accuracies, f1s)
        important_features = self.counter.most_common()[:20]
        important_features_df = pd.DataFrame.from_records(important_features, columns=["feature_name", "count"])
        return important_features_df,results


    def generate_text_result(self, A_precisions, A_recalls, K_precisions, K_recalls, mean_aucs,accuracies, f1s):
        text1= f"{self.experiment_name}\n"
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
