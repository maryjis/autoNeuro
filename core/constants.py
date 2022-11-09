GRID_CONFIG_MODELS ={
    'xgb': {
        'model__use_label_encoder': [False],
        'model__objective': ['binary:logistic'],
        'model__eval_metric': ['error'],
        'model__eta': [0.3, 0.001, 0.1],
        'model__max_depth':  [6,7,3,10],
        'model__scale_pos_weight': [1,0.7, 0.3],
        'model__subsample': [1, 0.5],
        'model__alpha': [0, 0.1, 1],
        'model__random_state': [42],
   },
   "svm": {
        'model__C': [0.01,0.1,1,3,10],
        'model__kernel':['linear', 'poly', 'rbf', 'sigmoid'],
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
