GRID_CONFIG_MODELS = {
    'lgb': {
        'model__boosting_type': ['gbdt'],
        'model__objective': ['binary'],
        'model__n_estimators': [50],
        'model__num_leaves': [6],
        'model__max_depth': [3],
        'model__learning_rate': [0.05, 0.1],

        'model__class_weight': [None],
        'model__min_split_gain': [0.3],
        'model__min_child_samples': [5],

        'model__subsample': [0.2, 0.5],
        'model__colsample_bytree': [0.2, 0.5],

        'model__reg_alpha': [0.0],
        'model__reg_lambda': [0.0],

        'model__random_state': [1380],
        'model__n_jobs': [1],
        'model__silent': [True],
        'model__importance_type': ['split'],
    },
    'xgb': {
        'model__use_label_encoder': [False],
        'model__objective': ['binary:logistic'],
        'model__eval_metric': ['error'],
        'model__eta': [0.1],
        'model__max_depth':  [6,7,3,10],
        'model__scale_pos_weight': [1],
        'model__subsample': [1, 0.5],
        'model__alpha': [0],
        'model__random_state': [42],
   },
   "svm": {
        'model__C': [0.01,0.1,1,3,10],
        'model__kernel':['linear', 'rbf'],
        'model__class_weight': ["balanced", None],
        'model__probability': [True],
        'model__random_state': [42],
    },
    "rf": {
        'model__n_estimators': [10, 50,100,500],
        'model__max_depth':  [3,5,7,9],
        'model__class_weight': ["balanced", "balanced_subsample", None],
        'model__random_state': [42],
    },
   "lr": {
        'model__class_weight': ["balanced", None],
        'model__penalty': ['l1', 'l2'],
        'model__C': [0.01,0.1,1,3,10],
        'model__solver' : ['liblinear'],
        'model__random_state': [42],
    }
}
