# Run an XGBoost model with hyperparmaters that are optimized using hyperopt
# The output of the script are the best hyperparmaters
# The optimization part using hyperopt is partly inspired from the following script: 
# https://github.com/bamine/Kaggle-stuff/blob/master/otto/hyperopt_xgboost.py


# Data wrangling

import pandas as pd

# Scientific 

import numpy as np
import random

import Train
from tqdm import tqdm
import matplotlib.pyplot as plt

# Machine learning

import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, accuracy_score, classification_report

# Hyperparameters tuning

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
import pickle

import sys

# Some constants

SEED = 314159265
VALID_SIZE = 0.2
TARGET = 'outcome'

#-------------------------------------------------#

# Scoring and optimization functions


def score(params):
    #print("Training with params: ")
    #print(params)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = CLP.xg_train #xgb.DMatrix(train_features, label=y_train)
    dvalid = CLP.xg_test #xgb.DMatrix(valid_features, label=y_valid)
    #watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgb.train(params, dtrain, num_round,
                          #evals=watchlist,
                          verbose_eval=True)
    predictions = gbm_model.predict(dvalid,
                                    ntree_limit=gbm_model.best_iteration)
    #print(predictions)
    loss = log_loss(y_valid, predictions)
    full_valid = CLP._create_full_class(y_valid)
    full_pred = CLP._create_full_class(np.argmax(predictions, axis=1))
    # TODO: Add the importance for the selected features
    #print("Score {0}".format(loss))
    #print("Accuracy {}".format(accuracy_score(y_valid, np.argmax(predictions, axis=1))))
    #print("Full accuracy {}".format(accuracy_score(full_valid, full_pred)))
    return {'loss': loss, 'status': STATUS_OK}

def testing(params):
    #print("Training with params: ")
    #print(params)
    #params['seed'] = random.randint(0, 1e6)
    num_round = int(params['n_estimators'])
    #del params['n_estimators']
    dtrain = CLP.xg_train #xgb.DMatrix(train_features, label=y_train)
    dvalid = CLP.xg_test #xgb.DMatrix(valid_features, label=y_valid)
    #watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgb.train(params, dtrain, num_round,
                          #evals=watchlist,
                          verbose_eval=True)
    predictions = gbm_model.predict(dvalid,
                                    ntree_limit=gbm_model.best_iteration)
    #print(predictions)
    loss = log_loss(y_valid, predictions)
    full_valid = CLP._create_full_class(y_valid)
    full_pred = CLP._create_full_class(np.argmax(predictions, axis=1))
    # TODO: Add the importance for the selected features
    print("Score {0}".format(loss))
    print("Accuracy {}".format(accuracy_score(y_valid, np.argmax(predictions, axis=1))))
    print("Full accuracy {}".format(accuracy_score(full_valid, full_pred)))
    print(classification_report(y_valid, np.argmax(predictions, axis=1)))
    print(classification_report(full_valid, full_pred))
    #return accuracy_score(y_valid, np.argmax(predictions, axis=1)), accuracy_score(full_valid, full_pred)

if __name__=="__main__":

    # Read in data wrt kic and fill etc.
    NDAYS = int(sys.argv[1]) #-1

    dataset = 'KIC'
    accounting_for_TESS = False #True
    df = pd.read_csv('Colours_New_Gaps_output_data_noise_'+str(dataset)+'_'+str(NDAYS)+'_APOKASC.csv')
    bad_kics = [10403036, 9846355, 5004660, 4936438, 6450613, 10597648, 5395942]
    df = df[~df.KIC.isin(bad_kics)] 

    # Set up training class
    CLP = Train.Train(df, dataset, NDAYS)
    # Preprocess
    CLP.preprocess_data()
    # Set up data for training
    CLP.setup_for_training(features=['KIC', 'numax', 'var', 'zc', 'hoc', 'abs_k_mag'])

    # Extract the train and valid (used for validation) dataframes from the train_df
    train_features, valid_features, y_train, y_valid = CLP.X_train.drop(['KIC', 'numax'], axis=1), CLP.X_test.drop(['KIC', 'numax'], axis=1), CLP.Y_train, CLP.Y_test
    print('The training set is of length: ', len(train_features.index))
    print('The validation set is of length: ', len(valid_features.index))

    #-------------------------------------------------#
    params = {
        'n_estimators': 696,
        'eta': 0.35,
        'max_depth': 10,
        'min_child_weight': 5.0,
        'subsample': 0.75,
        'gamma': 7.5,
        'colsample_bytree': 1.0,
        'eval_metric': 'mlogloss',
        'objective': 'multi:softprob',
        'num_class': 5,
        'nthread': 4,
        'silent': 1,
        'seed': SEED
    }

    params = {
        'n_estimators': 779.0,
        'eta': 0.45,
        'max_depth': 7,
        'min_child_weight': 3.0,
        'subsample': 0.9,
        'gamma': 7.5,
        'colsample_bytree': 0.9,
        'eval_metric': 'mlogloss',
        'objective': 'multi:softprob',
        'num_class': 5,
        'nthread': 16,
        'silent': 1,
        'seed': SEED
    }
    #full = np.zeros(100)
    #raw = np.zeros(100)
    #for i in tqdm(range(100), total=100):    
    #    raw[i], full[i] = testing(params)
    #print("RAW: ", np.mean(raw), np.std(raw))
    #print("Full: ", np.mean(full), np.std(full))
    #plt.hist(full, bins=15, histtype='step', density=True)
    #plt.hist(raw, bins=15, histtype='step', density=True)
    #plt.ylabel(r'Probability Density', fontsize=18)
    #plt.xlabel(r'Classification accuracy', fontsize=18)
    #plt.show()
    #sys.exit()
    # Run the optimization

    # Trials object where the history of search will be stored
    # For the time being, there is a bug with the following version of hyperopt.
    # You can read the error messag on the log file.
    # For the curious, you can read more about it here: https://github.com/hyperopt/hyperopt/issues/234
    # => So I am commenting it.
    trials = Trials()

    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        'max_depth':  scope.int(hp.quniform('max_depth', 1, 13, 1)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': 7.5,#hp.quniform('gamma', 0.5, 15, 0.5),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'mlogloss',
        'objective': 'multi:softprob',
        'num_class': 5,
        'nthread': 16,
        'silent': 1,
        'seed': SEED
    }
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest,
                trials=trials, 
                max_evals=250, show_progressbar=True)
    # The trials database now contains 100 entries, it can be saved/reloaded with pickle or another method
    pickle.dump(trials, open("hyperopt_"+str(NDAYS)+".pkl", "wb"))
    print("The best hyperparameters are: ", "\n")
    print(best)
    for key, value in best.items():
        if key == 'max_depth':
            params[key] = int(value)
        else:
            params[key] = value
    print(params)
    testing(params)
