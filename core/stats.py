import pandas as pd
from scipy.stats import ttest_ind, shapiro, levene, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt


class FeaturesStats():
    def __init__(self, dataset, important_features):
        self.dataset = dataset
        self.important_features = important_features
        self.important_features['p-value'] = ""
        self.important_features['test'] = ""
        self.important_features['normality'] = False

    def get_stats(self):
        print("______________stats___________________")
        for important_feature in self.important_features['feature_name'].tolist():
            print(important_feature)
            self.check_features(important_feature)

        print("_______________________________________________")
        self.important_features['p-value<0.05'] = self.important_features['p-value'] < 0.05

        return self.important_features

    def check_features(self, feature_column):
        res_0 = shapiro(self.dataset.loc[self.dataset["target"] == "Patient"][feature_column])
        res_1 = shapiro(self.dataset.loc[self.dataset["target"] == "Control"][feature_column])

        print("Test normality: ", res_0[1], res_1[1])
        a_feature = self.dataset.loc[self.dataset["target"] == "Patient", feature_column].values
        k_feature = self.dataset.loc[self.dataset["target"] == "Control", feature_column].values

        print("Patient: ", a_feature.mean())
        print("Control: ", k_feature.mean())

        if res_0[1] < 0.05 and res_1[1] < 0.05:
            res2 = levene(self.dataset.loc[self.dataset["target"] == "Patient"][feature_column],
                          self.dataset.loc[self.dataset["target"] == "Control"][feature_column])
            print("Test homogenius:", res2)

            if res2[1] < 0.05:
                t_test_stat = ttest_ind(a_feature, k_feature, equal_var=False)
                self.important_features.loc[self.important_features['feature_name'] == feature_column, "normality"] = True
                self.important_features.loc[self.important_features['feature_name'] == feature_column, "p-value"] = t_test_stat.pvalue
                self.important_features.loc[self.important_features['feature_name'] == feature_column, "test"] = "t-test"
            else:
                t_test_stat = mannwhitneyu(a_feature, k_feature)
                self.important_features.loc[
                    self.important_features['feature_name'] == feature_column, "p-value"] = t_test_stat.pvalue
                self.important_features.loc[
                    self.important_features['feature_name'] == feature_column, "test"] = "mannwhitneyu"
        else:
            t_test_stat = mannwhitneyu(a_feature, k_feature)
            self.important_features.loc[
                self.important_features['feature_name'] == feature_column, "p-value"] = t_test_stat.pvalue
            self.important_features.loc[
                self.important_features['feature_name'] == feature_column, "test"] = "mannwhitneyu"

        print(t_test_stat)
        sns.displot(self.dataset, x=feature_column, hue="target", kind="kde")
        plt.show()
