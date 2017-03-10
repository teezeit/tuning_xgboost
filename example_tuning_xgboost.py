# -*- coding: utf-8 -*-
"""
March 2016
tobias hoelzer
This script optimizes parameters for xgboost using greeedy gridsearch + crossvalidation
"""
import pandas as pd
import numpy as np
import tuning_xgboost
import xgboost as xgb
############################################################################

#1. create toy dataset 
header               = ['F1','F2','F3','F4','label']
df_train             = pd.DataFrame(np.random.rand(100,5),columns=header)
df_train['label']    = df_train['label'].apply(lambda x: 0. if x<0.5 else 1)


#2. define gridsearch parameters
#   a) model complexity
        #i)  max_depth
        #ii) min_child_weight
#   b) randomness
        #i)  subsample
        #ii) colsample_bytree
#   c) stepsize
        #i)  eta = n_estimators
        #ii) learning_rate
#   d) weighting positive data
        #i) scale_pos_weight

#parameter grid 1
n_estimators        = [50, 75, 100]
learning_rate       = [0.1, 0.2, 0.3]
param_grid_1        = dict(n_estimators=n_estimators,learning_rate=learning_rate)
#parameter grid 2
max_depth           = [3, 4]
min_child_weight    = [1, 2]
param_grid_2        = dict(max_depth=max_depth,min_child_weight=min_child_weight)
#paremeter grid 3
colsample_bylevel   = [0.7, 0.8, 0.9]
subsample           = [0.5, 0.6, 0.7]
param_grid_3        = dict(colsample_bylevel=colsample_bylevel, subsample=subsample)
#parameter grid 4
scale_pos_weight    = [1,2,3,4]
param_grid_4        = dict(scale_pos_weight=scale_pos_weight)


#3. prepare for search
X_train              = df_train.loc[:,header[:-1]].values
y_train              = df_train.loc[:,header[ -1]].values
hyperparameter_grids = [param_grid_1,param_grid_2,param_grid_3,param_grid_4]
booster              = xgb.XGBClassifier()

#4 a) Run tuning
print 'Run Simple Parameter Tuning\n________'
tuned_estimator      = tuning_xgboost.grid_search_tuning(X_train,y_train,hyperparameter_grids,booster)
tuned_parameters     = tuned_estimator.get_params()

for parameter in  tuned_parameters:
    print parameter, '\t\t',tuned_parameters[parameter]





#4 b).  additional parameters

print '\n\n Run Parameter Tuning with extra parameters\n________'
gridsearch_params = {
                    'cv'        : 2,
                    'scoring'   : 'roc_auc',#'roc_auc', 'average_precision'
                    'verbose'   : 2
                    }
tuned_estimator     = tuning_xgboost.grid_search_tuning(X_train,y_train,hyperparameter_grids,booster,gridsearch_params,verbose=False,plotting=True)
tuned_parameters    = tuned_estimator.get_params()

for parameter in  tuned_parameters:
    print parameter, '\t\t',tuned_parameters[parameter]



#end of file example_tuning_xgboost.py
