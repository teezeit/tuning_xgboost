# -*- coding: utf-8 -*-
"""
March 2016
author: teezeit
This script optimizes parameters for xgboost using greeedy gridsearch + crossvalidation
"""
# Imports
import numpy as np
import xgboost as xgb
import tuning_xgboost

#. define gridsearch parameters
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

#Training data + label
X_train		 = np.random.rand(100,4)
y_train		 = np.random.randint(2,size=100)

#Grids
grid1  		 = {'n_estimators' : [50, 100, 200], 'learning_rate' : [0.1, 0.2, 0.3]}
grid2 		 = {'max_depth' : [2,3,4], 'min_child_weight' : [4,5]}
grid3 		 = {'colsample_bylevel' : [0.7, 0.8, 0.9], 'subsample' : [0.5, 0.6, 0.7]}
grid4 		 = {'scale_pos_weight' : [1,2,5]}
hyperlist_to_try = [grid1, grid2,grid3,grid4]

#Booster
booster 	      = xgb.XGBClassifier()

###############################################################################
# Now run
print 'Run Simple Parameter Tuning\n________'
tuned_estimator   = tuning_xgboost.grid_search_tuning(X_train,y_train,hyperlist_to_try,booster)
tuned_parameters  = tuned_estimator.get_params()

for parameter in  tuned_parameters:
    print parameter, '\t\t',tuned_parameters[parameter]

# Define additional parameters

print '\n\n Run Parameter Tuning with extra parameters given to GridSearchCV\n________'
gridsearch_params = {
                    'cv'        : 2,
                    'scoring'   : 'roc_auc',#'roc_auc', 'average_precision'
                    'verbose'   : 2
                    }
tuned_estimator     = tuning_xgboost.grid_search_tuning(X_train,y_train,hyperlist_to_try,booster,gridsearch_params,verbose=False,plotting=True)
tuned_parameters    = tuned_estimator.get_params()

for parameter in  tuned_parameters:
    print parameter, '\t\t',tuned_parameters[parameter]



#end of file example_tuning_xgboost.py
