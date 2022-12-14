# autoNeuro
Pipeline to run the experiments with tabular data for classification tasks (Major Depressive Disorder vs Healthy Control, Schisophrenia vs Healthy Control).

Pipeline consist of several steps

1. **GridSearchBase** - search for the best ML method  with best feature selection algorithm from (XGBClassifier, SVC, RandomForestClassifier(),
   "lr" : LogisticRegression() ) and feature selection methods (SelectKBest, RandomForestClassifier (select top important features), LogisticRegression (select top important features) ).It is located in core/gridcv.py module. You can configure the parameters for grid search algorithm for each model in GRID_CONFIG_MODELS in core/constants.py module. 
2. **ExperimentsInfo** - calculate the most important features for the best methods,  build roc curves and other metrics, save important features as DataFrame  to  EXPERIMENTS_PATH (experiments.py module). It is located in core/metrics.py module.
3. **FeaturesStats**  - calculate the distributions and post-hoc t-test for the the most important features calculated in the previos step. It is located in core/stats.py
   
**How to run**

You can run code using experiments.py or see examples with jupyter notebooks in experiments package.

      X - pd.DataFrame with numeric data
      y - pd.Series with target values
      experiment_name - name of experiment (results are saved in directiry with this name) 

      best_result,best_f1 =run(X,y, experiment_name=file.stem, repeats =10, topN=1,scaling=False)





 
