# tuning_xgboost
Using consecutive (greedy) gridsearch with cross validation to tune xgboost hyperparameters in python. This is useful, when the hyperparameter space is too big.
I recommend using pairwise gridsearch.

All information and part of the code is taken from [Jesse Steinweg-Woods](https://jessesw.com/XG-Boost/), [Jason Brownlee](http://machinelearningmastery.com/tune-learning-rate-for-gradient-boosting-with-xgboost-in-python/) and [Aarshay Jain](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/). Thank you!

This is for anyone, who doesn't want to dig too deep into xgboost parameter tuning.

**Usage:**
'''
import tuning_xgboost
tuned_estimator      = tuning_xgboost.grid_search_tuning(X_train,y_train,hyperparameter_grids,booster)
tuned_parameters     = tuned_estimator.get_params()
'''

[example_tunign_xgboost.py](https://github.com/teezeit/tuning_xgboost/blob/master/example_tuning_xgboost.py) shows a minimal working example.




1. Define the hyperparameters in dictionaries that you want to tune
  * '''
  grid1 = {'n_estimators' : [50, 100, 200], 'learning_rate' : [0.1, 0.2, 0.3]}
  grid2 = {'max_depth' : [2,3,4], 'min_child_weight' : [4,5]}
  '''
2. Put them into a list
  * '''hyperlist = [grid1, grid2]'''
3. Define your booster
  * '''xgb.XGBClassifier()'''
4. Run the consecutive gridsearch
  * '''tuned_estimator      = tuning_xgboost.grid_search_tuning(X_train,y_train,hyperparameter_grids,booster)'''
  * The function will first run grid GridSearchCV on grid1, store the best hyperparameters then proceed to run GridSearchCV on grid2. The overall best booster is then returned by the function.




