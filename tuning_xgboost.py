########### IMPORTS ##########################################################

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV   #Perforing grid search
import matplotlib.pyplot as plt
#############  FUNCTION DEFINITIONS ##########################################
def plot_gridsearch(means,sterr,params,paramsnames,scoring,plot_subname='test',savefig=False,output_dir=None,plotfig=False):

    if len(params)>2: return

    fig = plt.figure()
    if len(params)==2:
        p1 = params[0]
        p2 = params[1]
        p1name = paramsnames[0]
        p2name = paramsnames[1]
        scores = np.reshape(means,(len(p1), len(p2)))
        yerrs  = np.reshape(sterr,(len(p1), len(p2)))
        for i, value in enumerate(p1):
            #plt.plot(p2, scores[i],'o-', label=p1name +': ' + str(value))
            plt.errorbar(p2, scores[i], yerr=yerrs[i], fmt='o-', label=p1name +': ' + str(value))
        plt.legend(loc='best')
        plt.xlabel(p2name)
        plt.ylabel(scoring)
    else:
        p1      = params[0]
        p1name  = paramsnames[0]
        scores  = np.array(means)
        yerrs   = np.array(sterr)
        #plt.plot(p1,scores,'o-',label=p1name)
        plt.errorbar(p1, scores, yerr=yerrs, fmt='o-', label=p1name)
        plt.xlabel(p1name)
        plt.ylabel(scoring)
        
    plt.grid(True)
    if (savefig and output_dir): plt.savefig(output_dir+'gridsearch_'+plot_subname+'.png',dpi=300)
    if plotfig: plt.show()

def analyse(grid_result, grid_hyper_parameter,gridsearch_params,plotlabel,verbose,plotting):
    n_folds = gridsearch_params['cv']
    scoring = gridsearch_params['scoring']
    if verbose:
        print '_________________\n\nFinished Grid Search:'
        print 'trying: ', grid_hyper_parameter
        print '\nBest: {:.6f} using {}'.format(grid_result.best_score_, grid_result.best_params_)

    #get results of grid search
    means   = grid_result.cv_results_['mean_test_score']
    stds    = grid_result.cv_results_['std_test_score']
    params  = grid_result.cv_results_['params']
   
    if verbose:
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    
    # plot results   
    sterrs = np.divide(stds,n_folds) #transform standard deviation into standard errors
    if plotting: plot_gridsearch(means,sterrs,grid_hyper_parameter.values(),grid_hyper_parameter.keys(),scoring,plot_subname='grid_'+str(plotlabel)+'_',output_dir='./',savefig=True)

def grid_search_tuning(X,y,hyperparameter_grids,booster,
                       gridsearch_params    = {'cv':5,'scoring':'average_precision'},
                       verbose              = True,
                       plotting             = True):
    '''
    Performs consecutive (greedy) GridsearchCV over list of dictionaries of hyperparameters.
    
    Parameters
    ----------
    X :  array-like, shape = [n_samples, n_features].
        Training vector, where n_samples is the number of samples and n_features is the number of features.
        
        
    y : array-like, shape = [n_samples] or [n_samples, n_output]
        Target relative to X for classification or regression.
        
        
    hyperparameter_grid : list of dictionaries - [{Grid_Dictionary_1}, {Grid_Dictionary_2}, ...]
        Grid_Dictionary_i = {key_of_hyperparameter : [list of values to try]}
        
        e.g.: hyperparameter_grid = [{'learning_rate': [0.1, 0.2, 0.3], 'n_estimators': [50, 75, 100]},{'max_depth': [3, 4], 'min_child_weight': [1, 2]}, {'colsample_bylevel': [0.7, 0.8, 0.9], 'subsample': [0.5, 0.6, 0.7]},{'scale_pos_weight': [1, 2, 3, 4]}]
    
    
    booster : xgboost Booster Object
        e.g. xgboost.XGBClassifier()
    
    
    gridsearch_params : additional gridsearch parameters,default {'cv':5,'scoring':'average_precision'}
        This is fed to GridsearchCV()
                      
    verbose : bool, default=True
        Printing Gridsearch Results to Screen
    
    plotting : bool, default=True
        Plotting Gridsearch Results
    
    
    '''
    best_estimator = booster

    for run,hyperparameter_grid in enumerate(hyperparameter_grids):        
        #gridsearch params                 
        gridparams          = {'estimator'     : best_estimator,
                               'param_grid'    : hyperparameter_grid,
                               }
        gridparams.update(gridsearch_params) # add more gridsearch parameters
        
        # Fit Gridsearch
        grid_result     = GridSearchCV(**gridparams).fit(X,np.squeeze(y))
        #save best result
        best_estimator  = grid_result.best_estimator_
        #maybe show
        if verbose or plotting:
            analyse(grid_result,hyperparameter_grid,gridparams,run,verbose,plotting)
    
    return best_estimator


#end of file tuning_xgboost.py
