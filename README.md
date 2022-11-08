# autoNeuro
Pipeline to run the experiments with tabular data for classification tasks (Major Depressive Disorder vs Healthy Control, Schisophrenia vs Healthy Control).

Pipeline consist of several steps

1. GridSearchBase - search for the best ML method  with best feature selection algorithm from ( "xgb": XGBClassifier(), "svm": SVC(), "rf" : RandomForestClassifier(),
   "lr" : LogisticRegression() ) and feature selection methods (SelectKBest, RandomForestClassifier (select top important features), LogisticRegression (select top important features) ).It is located in gridcv.py module. You can configure the parameters for grid search algorithm for each model in GRID_CONFIG_MODELS in constants.py module. 
2. ExperimentsInfo - calculate the most important features for the best methods,  build roc curves and other metrics, save important features as DataFrame  to  EXPERIMENTS_PATH (experiments.py module). It is located in metrics.py module.
3. FeaturesStats  - calculate the distributions and post-hoc t-test for the the most important features calculated in the previos step.
   
