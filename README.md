# tuning_xgboost

## Minimal Example
```
import numpy as np
import xgboost as xgb
import tuning_xgboost
#
X_train		 	= np.random.rand(100,4)
y_train		 	= np.random.randint(2,size=100)
grid1  		 	= {'n_estimators' : [50, 100, 200], 'learning_rate' : [0.1, 0.2, 0.3]}
grid2 		 	= {'max_depth' : [2,3,4], 'min_child_weight' : [4,5]}
grid3 		 	= {'colsample_bylevel' : [0.7, 0.8, 0.9], 'subsample' : [0.5, 0.6, 0.7]}
grid4 		 	= {'scale_pos_weight' : [1,2,5]}
hyperlist_to_try 	= [grid1, grid2,grid3,grid4]
booster 	 	= xgb.XGBClassifier()

tuned_estimator  = tuning_xgboost.grid_search_tuning(X_train,y_train,hyperlist_to_try,booster)
tuned_parameter  = tuned_estimator.get_params()
#you can also pass more parameters which will be fed into GridSearchCV().
```
* *grid_search_tuning* will first run grid GridSearchCV on grid1, store the best hyperparameters, then proceed to run GridSearchCV on grid2, store the best parameters from grid1 and grid2, then proceed to run GridSearchCV on grid3, and so on. The overall best booster be returned in the end.
* If you want, the parameters and the results will be plotted (if you don't optimize more than 2 at the same time)

## What is this for?
[tuning_xgboost.py](https://github.com/teezeit/tuning_xgboost/blob/master/tuning_xgboost.py) helps you using consecutive (greedy) gridsearch with cross validation to tune xgboost hyperparameters in python. This is useful, when the hyperparameter space is too big or you don't really want to dig too deep into how this all works.
I recommend using pairwise gridsearch. If you have no idea what you are doing, use those pairs:

1. pair 1 -stepsize
  * eta = n_estimators
  * learning_rate
2. pair 2 - model complexity
  * max_depth
  * min_child_weight
3. pair 3 - randomness
  * subsample
  * colsample_bytree
4. scaling of positive data
  * scale_pos_weight

Just define a list of values you like to try and give it to tuning_xgboost.grid_search_tuning(). It returns the trained booster with optimized hyperparameters. If you want it also prints the Status on the Screen and Plots the Results. Awesome :-).

All information and part of the code is taken from [Jesse Steinweg-Woods](https://jessesw.com/XG-Boost/), [Jason Brownlee](http://machinelearningmastery.com/tune-learning-rate-for-gradient-boosting-with-xgboost-in-python/) and [Aarshay Jain](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/). Thank you!

[example_tuning_xgboost.py](https://github.com/teezeit/tuning_xgboost/blob/master/example_tuning_xgboost.py) shows more working examples.






