#!/usr/bin/env/ python3

import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
import sys
import xgboost as xgb

from operator import itemgetter
from scipy import interp
from scipy.stats import randint as sp_randint
from sklearn import mixture
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, ShuffleSplit, train_test_split
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from xgboost import XGBClassifier
from utilities import *
# Set seed to ensure reproducible results!
np.random.seed(0)
random.seed(0)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dfstar = pd.read_csv('New_Gaps_output_data_noise_KIC_-1_fullKep.txt')#-1.txt')#80.txt')
    print(dfstar.head())
    dfstar = dfstar[dfstar['fill'] > 0.5]
    dfstar = dfstar[dfstar['ndata'] > 1000]
    #print(np.min(dfstar['ndata']))
    #sys.exit()
    dr25 = pd.read_csv('dr25catalogue.txt', delimiter=r'\s+')
    dfstar = pd.merge(dfstar, dr25[['KIC', 'Teff']], how='inner', on='KIC')
    hot = dfstar[dfstar['Teff'] > 6500]
    hot['det'] = 0
    hot = hot[['KIC', 'det']]
    print(dr25.head())
    rg = pd.read_csv('yu_2016.csv', delimiter='|', na_values='--')
    rg['det'] = 1
    rg.rename(columns={'KICID': 'KIC'}, inplace=True)
    rg = rg[['KIC', 'det']]
    nondet = pd.read_csv('Non-Det.txt')
    nondet['det'] = 0
    print(len(rg), len(nondet))
    rg = rg.append(nondet, ignore_index=True)
    rg = rg.append(hot, ignore_index=True)
    training = pd.merge(dfstar, rg, how='inner', on='KIC')
    predict_df = dfstar[(~dfstar['KIC'].isin(training['KIC']))]

    print(len(training)+len(predict_df))

     # Only train classifier on best 3 features
    X_predict = predict_df[['KIC', 'var', 'zc', 'hoc', 'mc', 'faps', 'Teff']]

    X = training[['KIC', 'var', 'zc', 'hoc', 'mc', 'faps', 'Teff']]

    y = training['det'].astype(int)

    print(np.unique(y))
    print("Number of classes: {}".format(len(np.unique(y))))
   
    # Train, test split
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        stratify=y)
    print(np.shape(X_predict), np.shape(Y_test))

    # Create Dmatrix for use with xgboost
    xg_train = xgb.DMatrix(X_train.drop(['KIC'], axis=1), label=Y_train.values)#, weight=X_train['weights'])
    xg_test =  xgb.DMatrix(X_test.drop(['KIC'], axis=1), label=Y_test.values)
    xg_predict = xgb.DMatrix(X_predict.drop(['KIC'], axis=1))

    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'binary:logistic'
    param['eval_metric'] = 'logloss'
    # scale weight of positive examples
    #param['eta'] = 0.1
    #param['max_depth'] = 3
    #param['min_child_weight'] = 4
    #param['subsample'] = 0.7
    #param['colsample'] = 0.7
    param['silent'] = 1
    param['nthread'] = 4
    #param['num_class'] = len(np.unique(y))
    param['seed'] = 0

    # Set watchlist and number of rounds
    watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
    num_round = 1000
    """
    param_test1 = {
     'max_depth': np.arange(3,10,2),
     'min_child_weight': np.arange(1,6,2)
    }
    print(param_test1)
    gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
	 						min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
	 						objective= 'multi:softprob', nthread=4, scale_pos_weight=1, seed=27),
	 						param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=5)
    gsearch1.fit(X_train, Y_train)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    sys.exit()
    """
    # Progress dictionary
    progress = {}
    # Cross validation - stratified with 10 folds, preserves class proportions
    cvresult = xgb.cv(param, xg_train, num_round, stratified=True, nfold=10, seed = 0,
                      callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
    print(cvresult)
    nrounds = np.linspace(1, num_round+1, num_round)

    # Plot results of simple cross-validation
    plt.errorbar(nrounds, cvresult['train-logloss-mean'], yerr=cvresult['train-logloss-std'], fmt='o', color='b', label='Training error')
    plt.errorbar(nrounds, cvresult['test-logloss-mean'], yerr=cvresult['test-logloss-std'], fmt='o', color='r', label='Test error')
    plt.show()


    # do the same thing again, but output probabilities
    #param['objective'] = 'multi:softprob'
    param['seed'] = 0
    # Number of rounds used that minimises test error in cross-validation used for last training of model
    bst = xgb.train(param, xg_train, int(cvresult[cvresult['test-logloss-mean'] == np.min(cvresult['test-logloss-mean'])].index[0]), watchlist, early_stopping_rounds=100, evals_result=progress)
    #int(cvresult[cvresult['test-mlogloss-mean'] == np.min(cvresult['test-mlogloss-mean'])].index[0])
    #bst = xgb.train(param, xg_train, int(cvresult[cvresult['test-merror-mean'] == np.min(cvresult['test-merror-mean'])].index[0]), watchlist, early_stopping_rounds=50, evals_result=progress)
    print(progress)
    # Print test and train errors
    #plt.plot(progress['train']['mlogloss'], color='b')
    #plt.plot(progress['test']['mlogloss'], color='b')
    #plt.plot(progress['train']['merror'], color='b')
    #plt.plot(progress['test']['merror'], color='b')
    #plt.show()
    # explain the model's prediction using SHAP values on the first 1000 training data samples

    # Make predictions
    yprob = bst.predict( xg_test )#.reshape( Y_test.shape[0], param['num_class'] )
    ypred = bst.predict( xg_predict )
    class_names = ['0', '1']

    # Get labels from class probabilities

    ylabel = np.round(yprob).astype(int)
    ypred_label = np.round(ypred).astype(int)

    print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != Y_test.as_matrix()[i] for i in range(len(Y_test))) / float(len(Y_test))))
    print ('predicting, classification accuracy= ', 1.0-(sum( int(ylabel[i]) != Y_test.as_matrix()[i] for i in range(len(Y_test))) / float(len(Y_test))))
    misc = np.array([ylabel[i] != Y_test.as_matrix()[i] for i in range(len(Y_test))]).ravel()
    print("Number of misclassified: {0} out of {1}".format(sum( int(ylabel[i]) != Y_test.as_matrix()[i] for i in range(len(Y_test))), len(Y_test)))
    y = np.bincount(Y_test.as_matrix().ravel()[misc])
    ii = np.nonzero(y)[0]
    print(zip(ii, y[ii]))
    print("Matthew's correlation coefficient: ", matthews_corrcoef(Y_test.as_matrix().ravel(), ylabel))
    # Binarize Labels
    from sklearn.preprocessing import label_binarize
    #y = label_binarize(Y_test, classes=[0,1])
    # Compute auc/roc etc.
    #fpr = dict()
    #tpr = dict()
    #roc_auc = dict()
    #n_classes = 2
    #for i in range(n_classes):
    #    fpr[i], tpr[i], _ = roc_curve(y[:,i], yprob[:,i])
    #    roc_auc[i] = auc(fpr[i], tpr[i])

    #classes = ['0', '1']
    #for i in range(n_classes):
    #    plt.plot(fpr[i], tpr[i], lw=2,
    #    label=r'ROC curve of class {} (area = {:0.3f})'.format(classes[i], roc_auc[i]))
    #plt.plot([0,1], [0,1], 'k--', lw=2)
    #plt.xlim(0, 1)
    #plt.ylim(0, 1.05)
    #plt.xlabel(r'False Positive Rate', fontsize=18)
    #plt.ylabel(r'True Positive Rate', fontsize=18)
    #plt.legend(loc='lower right')
    #plt.show()
    X_predict[ypred_label == 1].to_csv('New_osc.csv', index=False)
    print("Number of new solar-like oscillators: {}".format(len(ypred_label[ypred_label == 1])))
