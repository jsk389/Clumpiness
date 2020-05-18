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
import scipy
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
from plot_utilities import *
import operator
# Set seed to ensure reproducible results!
np.random.seed(0)
random.seed(0)

def separate_data(df, NDAYS):
    # Only train on stars that have enough data for respective classifier
    print(df.head())
    if NDAYS == -1:
        predict_df = df[(df['ndata'] < ((180 * 86400.0)/(29.4*60))) | (df['evo'] == 6)]
        predict_df = predict_df[predict_df['fill'] > 0.5]
        df = df[df['ndata'] > ((180 * 86400.0)/(29.4*60))]
        df = df[df['fill'] > 0.5]
    elif NDAYS == 180:
        predict_df = df[(df['ndata'] < ((80 * 86400.0)/(29.4*60))) | (df['evo'] == 6)]
        df = df[df['ndata'] > ((80 * 86400.0)/(29.4*60))]
        df = df[df['fill'] > 0.5]
        predict_df = predict_df[predict_df['fill'] > 0.5]
    elif NDAYS == 80:
        predict_df = df[(df['ndata'] < ((27 * 86400.0)/(29.4*60))) | (df['evo'] == 6)]
        df = df[df['ndata'] > ((27 * 86400.0)/(29.4*60))]
        df = df[df['fill'] > 0.5]
        predict_df = predict_df[predict_df['fill'] > 0.5]
    elif NDAYS == 27:
        predict_df = df[(df['evo'] == 6)]
        df = df[df['ndata'] < ((27 * 86400.0)/(29.4*60))]
        df = df[df['fill'] > 0.5]
        predict_df = predict_df[predict_df['fill'] > 0.5]
    else:
        pass
    return df, predict_df

def preprocess_Kepler(df):

    df['KIC'] = df['KIC'].astype(int)
    #print(len(df))
    KICS = df['KIC'][df['evo'] == 6].values.astype(int)
    all_kics = pd.read_csv('/home/kuszlewicz/Dropbox/Python/Clumpiness/apokasc_DR01.dat', delimiter=r'\s+')
    idxs = all_kics['KIC'].isin(KICS)
    tmp = all_kics[idxs]
    # Any star with numax > 180 is definitely going to be RGB
    tmp = tmp[tmp['numax'] > 180]
    tmp['KIC'] = tmp['KIC'].astype(int)

    df['evo'][df['KIC'].isin(tmp['KIC'])] = 0

    # Bad stars from Keaton
    bad_kics = np.loadtxt('../Tables/Bad_dwarfs_from_Keaton.txt')
    df = df[~df['KIC'].isin(bad_kics)]


    print(len(df))

    print(len(df))
    print(np.unique(df['evo'], return_counts=True))
    return df

if __name__=="__main__":

    # Read in data wrt kic and fill etc.
    NDAYS = 180
    dataset = 'KIC'
    accounting_for_TESS = False #True
    #df = pd.read_csv('New_Gaps_output_data_noise_'+str(dataset)+'_'+str(NDAYS)+'_APOKASC.csv')
    df = pd.read_csv('New_Gaps_MAD_MAD_output_data_noise_'+str(dataset)+'_'+str(NDAYS)+'_APOKASC.csv')
    bad_kics = [10403036, 9846355, 5004660, 4936438, 6450613, 10597648]
    df = df[~df.KIC.isin(bad_kics)]
    #print(np.shape(df), np.shape(df2))
    print(df.loc[(df['evo'] == 4) & (df['var'] > 2e4) & (df['var'] < 3e5) & (df['zc']  > 0.1) & (df['zc'] < 0.4), ])
    #print(df2.loc[(df2['evo'] < 4) & (df2['var'] < 1e2), ])
    #plt.plot(df['numax'], df['abs_k_mag'], '.')
    #plt.scatter(df.loc[df['evo'] == 4,'var'], df.loc[df['evo'] == 4,'zc'], c=df.loc[df['evo'] == 4,'evo'], marker='.')
    #plt.show()

#    df['var'] = df['var']**2
#    df['mc'] = df['mc']**2
#    df = pd.read_csv('New_Gaps_output_data_noise_KIC_-1_APOKASC.csv')
    print(df.head())
    print(np.shape(df))
    df['running_var'] = np.log10(np.abs(df['running_var']))
    #print(df['running_var'])
    if NDAYS == -1:
        df = df[df['running_var'] < 2.5]

    if dataset == 'KIC':
        df = preprocess_Kepler(df)

    # Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)
    #print(df.head())
    #sys.exit()
    #df['evo'] = df['evo'].map({'[ 0.]': 0,
    #                           '[ 1.]': 1,
    #                           '[ 2.]': 2,
    #                           '[ 3.]': 3,
    #                           '[ 4.]': 4,
    #                           '[ 6.]': 6})#.astype(int)

    df, predict_df = separate_data(df, NDAYS)

    print("LENGTH: ", len(df), len(predict_df))

    # Only train classifier on best 3 features
    X_predict = predict_df[['KIC', 'var', 'zc', 'hoc', 'mc', 'abs_k_mag']]##, 'faps', ]]

    df = df[df['evo'] < 5]
    print(df)
    # Data
    X = df[['KIC', 'numax', 'var', 'zc', 'hoc', 'mc', 'abs_k_mag']]##, 'faps', ]]
    print(X.tail())

    # Labels
    y = df['evo'].astype(int)
    print(np.unique(y))
    print("Number of classes: {}".format(len(np.unique(y))))


    # To account for redder passband in TESS
    if (NDAYS == 27) & (accounting_for_TESS == True):
        X['var'] *= (0.85**2)
        X['mc'] *= (0.85**2)

    #X['rand'] = np.random.normal(0, 1, len(X))
    #X['y'] = df['evo'].astype(int)
    #X.to_csv('random_feature_test.csv', index=False)
    #X = X.drop(['y'], axis=1)
    # Train, test split
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        stratify=y)

    # Create Dmatrix for use with xgboost
    xg_train = xgb.DMatrix(X_train.drop(['KIC', 'numax'], axis=1), label=Y_train)
    print(np.shape(X_train.drop(['KIC', 'numax'], axis=1)))
    xg_test =  xgb.DMatrix(X_test.drop(['KIC', 'numax'], axis=1), label=Y_test)#, weight=X_test['weights'])
    xg_predict = xgb.DMatrix(X_predict.drop(['KIC'], axis=1))
    #xg_test2 = xgb.DMatrix(X_test2.drop(['KIC'], axis=1))

    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softprob'
    param['eval_metric'] = 'mlogloss'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['gamma'] = 7.5
    param['max_depth'] = 3
    param['min_child_weight'] = 4
    param['subsample'] = 0.7
    param['colsample'] = 0.7
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = len(np.unique(y))
    param['seed'] = 0

    # Set watchlist and number of rounds
    watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
    num_round = 1000

    # Progress dictionary
    progress = {}
    # Cross validation - stratified with 10 folds, preserves class proportions
    cvresult = xgb.cv(param, xg_train, num_round, stratified=True, nfold=10, seed = 0,
                      callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
    print(cvresult)
    nrounds = np.linspace(1, num_round+1, num_round)

    # Plot results of simple cross-validation
    plt.errorbar(nrounds, cvresult['train-mlogloss-mean'], yerr=cvresult['train-mlogloss-std'], fmt='o', color='b', label='Training error')
    plt.errorbar(nrounds, cvresult['test-mlogloss-mean'], yerr=cvresult['test-mlogloss-std'], fmt='o', color='r', label='Test error')
    plt.show()


    # do the same thing again, but output probabilities
    #param['objective'] = 'multi:softprob'
    param['seed'] = 0
    # Number of rounds used that minimises test error in cross-validation used for last training of model
    bst = xgb.train(param, xg_train, int(cvresult[cvresult['test-mlogloss-mean'] == np.min(cvresult['test-mlogloss-mean'])].index[0]), watchlist, early_stopping_rounds=100, evals_result=progress)
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
    print(bst.get_fscore())

    print("BEST ITER: ", bst.best_ntree_limit)
    # Since early-stopping used, save out best iteration
    np.savetxt('best_iter_'+str(NDAYS)+'.txt', np.array([bst.best_ntree_limit]))

    if accounting_for_TESS == True:
        bst.save_model(str(NDAYS)+'_TESS_passband.model')
    else:
        print('SAVED MODEL!')
        bst.save_model(str(NDAYS)+'MAD.model')

    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    feat_import = pd.DataFrame(importance, columns=['feature', 'fscore'])
    feat_import['fscore_normed'] = feat_import['fscore'] / np.sum(feat_import['fscore'])
    print(feat_import)

    # Make predictions
    yprob = bst.predict( xg_test )#.reshape( Y_test.shape[0], param['num_class'] )

    class_names = ['Hs low', 'Hs high', 'Hs conf', 'CHeB', 'Noise']

    # Get labels from class probabilities
    ylabel = np.argmax(yprob, axis=1)
    numax = X_test['numax'].values
    idx = np.isfinite(numax)
    idx &= (numax > 0)
    numax = numax[idx]
    yp = yprob[idx]

    bins = np.linspace(0, 283, 50)
    bin_pds  = scipy.stats.binned_statistic(
        x         = numax,
        values    = yp[:,2],
        statistic = 'mean',
        bins      = bins)
    bin_edges = bin_pds[1]
    bin_width = (bin_edges[1] - bin_edges[0])
    rgb = {"x": bin_edges[1:] - bin_width/2,
           "y": bin_pds[0]}

    bin_pds  = scipy.stats.binned_statistic(
        x         = numax,
        values    = yp[:,3],
        statistic = 'mean',
        bins      = bins)
    bin_edges = bin_pds[1]
    bin_width = (bin_edges[1] - bin_edges[0])
    rc = {"x": bin_edges[1:] - bin_width/2,
           "y": bin_pds[0]}

    print(np.unique(ylabel, return_counts=True))
    print(np.unique(Y_test, return_counts=True))
    plt.step(x=rgb["x"], y=rgb["y"], color='C0')
    plt.step(x=rc["x"], y=rc["y"], color='C1')
    plt.xlabel(r'$\nu_{\mathrm{max}}$', fontsize=18)
    plt.xlim(0, 280)
    plt.ylim(0, 1)
    plt.ylabel(r'Class Probability', fontsize=18)
    plt.show()
    #plt.scatter(X_test['numax'], yprob[:,2],
    #            s=2, c='C0', marker='.')
    #plt.scatter(X_test['numax'], yprob[:,3],
    #                        s=2, c='C1', marker='.')
    #plt.xlabel(r'$\nu_{\mathrm{max}}$', fontsize=18)
    #plt.ylabel(r'Class Probability', fontsize=18)
    #plt.show()

    print(np.unique(ylabel, return_counts=True))
    print(np.unique(Y_test, return_counts=True))
    plt.scatter(X_test['numax'], np.max(yprob, axis=1),
                s=2, c=ylabel, marker='.')
    plt.xlabel(r'$\nu_{\mathrm{max}}$', fontsize=18)
    plt.ylabel(r'Largest Class Probability', fontsize=18)
    plt.colorbar(ticks=range(5), label='Class')
    plt.show()

    numax = X_test['numax'].values
    idx = np.isfinite(numax)
    ylab = ylabel[idx]
    ytest = Y_test[idx]
    numax = numax[idx]
    bins = np.linspace(0, 280, 28)

    n1, bin_edges1 = np.histogram(numax[numax > 0][ylab[numax > 0] < 3], bins=bins)
    centre1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
    n2, bin_edges2 = np.histogram(numax[numax > 0][ylab[numax > 0] == 3], bins=bins)
    centre2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2
    n3, bin_edges3 = np.histogram(numax[numax > 0][ytest[numax > 0] < 3], bins=bins)
    centre3 = (bin_edges3[:-1] + bin_edges3[1:]) / 2
    n4, bin_edges4 = np.histogram(numax[numax > 0][ytest[numax > 0] == 3], bins=bins)
    centre4 = (bin_edges4[:-1] + bin_edges4[1:]) / 2

    plt.figure(1)
    plt.step(x=centre1, y=n1, color='C0')
    plt.step(x=centre3, y=n3, color='C2')
    plt.figure(2)
    plt.step(x=centre2, y=n2, color='C0')
    plt.step(x=centre4, y=n4, color='C2')
    plt.show()



    plt.figure(1)
    plt.plot(X_test['numax'][ylabel < 3], X_test['zc'][ylabel < 3], 'C0.')
    plt.figure(2)
    plt.plot(X_test['numax'][ylabel == 3], X_test['zc'][ylabel == 3], 'C1.')
    plt.show()



    #ylabel2 = np.argmax(bst.predict(xg_test2), axis=1)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_test, ylabel.astype(int))
    class_names = ['LLRGB', 'HLRGB', 'Confusion RGB', 'CHeB', 'Noise']
    np.set_printoptions(precision=3)
    #class_names = ['Hs', 'CHeB', 'Noise']
    plt.figure(1)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, without normalization')
    plt.show()

    print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != Y_test.as_matrix()[i] for i in range(len(Y_test))) / float(len(Y_test))))
    print ('predicting, classification accuracy= ', 1.0-(sum( int(ylabel[i]) != Y_test.as_matrix()[i] for i in range(len(Y_test))) / float(len(Y_test))))
    #print('classification accuracy with kmag nans= ', 1.0-(sum( int(ylabel2[i]) != Y_test2.as_matrix()[i] for i in range(len(Y_test2))) / float(len(Y_test2))))
    misc = np.array([ylabel[i] != Y_test.as_matrix()[i] for i in range(len(Y_test))]).ravel()
    print("Number of misclassified: {0} out of {1}".format(sum( int(ylabel[i]) != Y_test.as_matrix()[i] for i in range(len(Y_test))), len(Y_test)))
    y = np.bincount(Y_test.as_matrix().ravel()[misc])
    ii = np.nonzero(y)[0]
    print(zip(ii, y[ii]))
    print("Matthew's correlation coefficient: ", matthews_corrcoef(Y_test.as_matrix().ravel(), ylabel))
    # Binarize Labels
    from sklearn.preprocessing import label_binarize
    y = label_binarize(Y_test, classes=[0,1,2,3,4])
    # Compute auc/roc etc.
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 5
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:,i], yprob[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    classes = [r'Low RGB', 'High RGB', 'RGB conf', 'HeCB', 'Noise']
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
        label=r'ROC curve of class {} (area = {:0.3f})'.format(classes[i], roc_auc[i]))
    plt.plot([0,1], [0,1], 'k--', lw=2)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel(r'False Positive Rate', fontsize=18)
    plt.ylabel(r'True Positive Rate', fontsize=18)
    plt.legend(loc='lower right')
    plt.show()

    # Create output
    ytrain_label = np.argmax(bst.predict(xg_train), axis=1)
    X_train['full_state'] = ytrain_label.astype(int)
    X_train['Flag'] = 0

    def create_full_class(labels):
        new_labels = labels.copy()
        new_labels[(new_labels == 0)|(new_labels == 1)|(new_labels == 2)] = 0
        new_labels[(new_labels == 3)] = 1
        new_labels[(new_labels == 4)] = 2
        return new_labels

    new_train_label = create_full_class(ytrain_label)

    X_train['final_state'] = new_train_label

    X_test['full_state'] = ylabel.astype(int)
    X_test['final_state'] = create_full_class(ylabel.astype(int))
    #X_train = X_train[Y_train != 4]
    #X_test = X_test[Y_test != 4]
    X_test['Flag'] = 1


    ypred_prob = bst.predict(xg_predict)
    ypred_label = np.argmax(ypred_prob, axis=1)
    X_predict['full_state'] = ypred_label.astype(int)
    X_predict['final_state'] = create_full_class(ypred_label.astype(int))
    X_predict['Flag'] = 2

    # Compare final state to labels!
    Y_train_final = Y_train.copy()
    Y_test_final = Y_test.copy()
    #
    Y_train_final[(Y_train_final == 0)|(Y_train_final == 1)|(Y_train_final == 2)] = 0
    Y_train_final[(Y_train_final == 3)] = 1
    Y_train_final[(Y_train_final == 4)] = 2
    Y_test_final[(Y_test_final == 0)|(Y_test_final == 1)|(Y_test_final == 2)] = 0
    Y_test_final[(Y_test_final == 3)] = 1
    Y_test_final[(Y_test_final == 4)] = 2
    # Compute accuracy
    from sklearn.metrics import accuracy_score
    print("Training set full accuracy: {}".format(accuracy_score(Y_train_final,
                                                                 X_train['final_state'])))
    print("Test set full accuracy: {}".format(accuracy_score(Y_test_final,
                                                                 X_test['final_state'])))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_train_final, X_train['final_state'])
    class_names = ['RGB', 'HeCB', 'Noise']
    np.set_printoptions(precision=3)
    #class_names = ['Hs', 'CHeB', 'Noise']
    plt.figure(1)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
    plt.figure(2)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
    plt.show()
    cnf_matrix = confusion_matrix(Y_test_final, X_test['final_state'])

    plt.figure(1)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
    plt.figure(2)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
    plt.show()

    print('--------------------------------')
    print(np.unique(X_train['final_state'].values, return_counts=True))
    print(np.unique(X_test['final_state'].values, return_counts=True))
    print(np.unique(X_predict['final_state'].values, return_counts=True))
    print('--------------------------------')
    print(ypred_prob[X_predict['final_state'] == 2])
    print(X_predict['KIC'][X_predict['final_state'] == 2])


    plt.scatter(X_train['var'], X_train['zc'], marker='.', c=X_train['final_state'])
    plt.scatter(X_test['var'], X_test['zc'],  marker='.', c=X_test['final_state'])
    plt.scatter(X_predict['var'], X_predict['zc'], marker='x', c=X_predict['final_state'], zorder=2)
    plt.scatter(X_predict['var'][X_predict['final_state'] == 2],
                X_predict['zc'][X_predict['final_state'] == 2],
                marker='x', color='r', zorder=3)
    plt.xlabel(r'Variance (ppm$^{2}$)', fontsize=18)
    plt.ylabel(r'Normalised number of zero crossings', fontsize=18)
    #plt.colorbar(label=r'Final Evolutionary State')
    plt.show()

    # Misclassification
    print(np.shape(X_test), np.shape(Y_test))
    df = X_test.copy()
    df['evo'] = Y_test.values
    df['classification'] = [int(int(ylabel[i]) != Y_test.values[i]) for i in range(len(Y_test))]
    df.to_csv('misclass_'+str(NDAYS)+'.csv', index=False)

    X_train = X_train[Y_train_final < 2]
    X_test = X_test[Y_test_final < 2]
    frames = [X_train, X_test, X_predict]
    result = pd.concat(frames)
    print(len(X_train), len(X_test), len(X_predict), len(result))
    cols = list(result)
    cols.insert(0, cols.pop(cols.index('KIC')))
    result = result.ix[:,cols]
    result = result.drop(['hoc', 'mc', 'var', 'zc', 'abs_k_mag'], axis=1)
    print(result.head())
    print(len(result))
    print(len(result)/6661)
    print(result.head())
    print(np.unique(result['Flag'], return_counts=True))
    print(np.unique(result['final_state'], return_counts=True))
    print(np.shape(result))
    #result.to_csv('Updated_TimeDomain_class_for_Yvonne.csv', index=False)
    #if accounting_for_TESS == True:
    #    bst.save_model(str(NDAYS)+'_TESS_passband.model')
    #else:
    #    bst.save_model(str(NDAYS)+'_MAD.model')


    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_test, ylabel)
    np.set_printoptions(precision=3)
    #class_names = ['Hs', 'CHeB', 'Noise']
    plt.figure(1)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
    plt.figure(2)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
    plt.show()

    # Save model
    #sys.exit()
    # Retrieve whether data corrrectly classified or not

    #data = X_test, Y_test, [int(ylabel[i]) != Y_test.values[i] for i in range(len(Y_test))]
    sys.exit()
    print(df)


    #bst.get_fscore()
    #mapper = {'f{0}'.format(i): v for i, v in enumerate(xg_train.feature_names)}
    #mapped = {mapper[k]: v for k, v in bst.get_fscore().items()}

    #xgb.plot_importance(mapped, color='red')
    #plt.show()

    #pd.DataFrame(bst.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)
    #plt.show()
    # Get feature importance
    #from collections import OrderedDict
    #import re
    #my_fun = lambda k,v: [k, int(v)]
    #b = OrderedDict(sorted(bst.get_score().items(), key=lambda t: my_fun(*re.match(r'([a-zA-Z]+)(\d+)',t[0]).groups())))
    import json
    with open('full_features.json', 'w') as f:
        f.write(json.dumps(bst.get_score()))
    # Plot feature importance
    xgb.plot_importance(bst)
    plt.show()
