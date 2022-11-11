import pandas as pd
from scipy.stats import ttest_ind, shapiro, levene, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt


class FeaturesStats:
    def __init__(self, dataset, important_features):
        self.dataset = dataset
        self.important_features = important_features
        self.important_features['p-value'] = ""
        self.important_features['test'] = ""
        self.important_features['normality'] = False

    def get_stats(self, plot_density=True):
        print("______________stats___________________")

        for important_feature in self.important_features['feature_name'].tolist():
            print(important_feature)
            self.check_features(important_feature, plot_density=plot_density)

        print("_______________________________________________")

        self.important_features['p-value<0.05'] = self.important_features['p-value'] < 0.05

        return self.important_features

    def check_features(self, feature_column: str, plot_density: bool = True):
        a_feature = self.dataset.loc[self.dataset["target"] == "Patient", feature_column]
        k_feature = self.dataset.loc[self.dataset["target"] == "Control", feature_column]

        # shapiro test between for two groups
        res_0 = shapiro(a_feature)
        res_1 = shapiro(k_feature)
        print("Test normality p-values: ", res_0[1], res_1[1])

        # mean for each column
        print("Patient: ", a_feature.mean())
        print("Control: ", k_feature.mean())

        # if feature is normal for both groups
        if res_0[1] >= 0.05 and res_1[1] >= 0.05:
            # test for equal variances
            res2 = levene(a_feature, k_feature)
            print("Equal variances test p-value:", res2)

            # vars are not equal
            if res2[1] < 0.05:
                t_test_stat = ttest_ind(a_feature, k_feature, equal_var=False)
                self.important_features.loc[self.important_features['feature_name'] == feature_column, "normality"] = True
                self.important_features.loc[self.important_features['feature_name'] == feature_column, "p-value"] = t_test_stat.pvalue
                self.important_features.loc[self.important_features['feature_name'] == feature_column, "test"] = "t-test"
            # vars are equal
            else:
                t_test_stat = ttest_ind(a_feature, k_feature, equal_var=True)
                #t_test_stat = mannwhitneyu(a_feature, k_feature)
                self.important_features.loc[
                    self.important_features['feature_name'] == feature_column, "p-value"] = t_test_stat.pvalue
                self.important_features.loc[
                    self.important_features['feature_name'] == feature_column, "test"] = "t-test"
        else:
            t_test_stat = mannwhitneyu(a_feature, k_feature)
            self.important_features.loc[
                self.important_features['feature_name'] == feature_column, "p-value"] = t_test_stat.pvalue
            self.important_features.loc[
                self.important_features['feature_name'] == feature_column, "test"] = "mannwhitneyu"

        print(t_test_stat)

        # just plot feature prob densitty
        if plot_density:
            sns.displot(self.dataset, x=feature_column, hue="target", kind="kde", common_norm=False)
            plt.show()
