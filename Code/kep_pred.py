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

from tqdm import tqdm as tqdm
# Set seed to ensure reproducible results!
np.random.seed(0)
random.seed(0)


if __name__=="__main__":

    # Read in data wrt kic and fill etc.
    #fill = np.loadtxt('n_data_kics.txt')
    # Read in data + preprocessing
    #data = np.loadtxt('New_output_data_noise_27.txt')
    #new_data = np.c_[fill, data]
    #df = pd.DataFrame(new_data,
    rg = pd.read_csv('yu_2016.csv', delimiter='|', na_values='--')
    print(list(rg.columns))
    rg.rename(columns={'KICID': 'KIC'}, inplace=True)
    #rgkics = rg[['KICID']]
    NDAYS = 80#27
    dataset = 'KIC'
    df = pd.read_csv('New_Gaps_output_data_noise_KIC_-1_APOKASC.txt')
    dfpred = pd.read_csv('New_Gaps_output_data_noise_KIC_-1_fullKep.txt')#-1.txt')#80.txt')

    dfpred = pd.merge(dfpred, rg[['KIC']], how='inner', on='KIC')

    fig, ax = plt.subplots()
    plt.scatter(df['zc'], df['hoc'], marker='.', zorder=2, alpha=0.1)
    #plt.scatter(dfstar2['zc'], dfstar2['hoc'], marker='.', alpha=0.1)
    plt.scatter(dfpred['zc'], dfpred['hoc'], marker='.', zorder=1)
    #plt.scatter(dfstar4['zc'], dfstar4['hoc'], marker='.', alpha=0.1)
    plt.show()

    plt.scatter(df['zc'], df['hoc'], marker='.')
    #plt.colorbar()
    print(sum(df['KIC'].isin(rg['KICID'])))
    print(sum(df['KIC'].isin(dfpred['KIC'])))

    plt.scatter(df['zc'][df['KIC'].isin(rg['KICID'])], df['hoc'][df['KIC'].isin(rg['KICID'])], marker='x', zorder=2, color='k', alpha=0.5)
    plt.axis('tight')
    plt.show()


    df = df.sample(frac=1).reset_index(drop=True)

    X = df[['KIC', 'var', 'zc', 'hoc', 'mc', 'faps']]
    #X = X[X['zc'] > 0]
    X['var'] = np.log10(X['var'])
    X = X[['var', 'zc', 'hoc']]


    # display predicted scores by the model as a contour plot
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    x, Y = np.meshgrid(x, y)
    XX = np.array([x.ravel(), Y.ravel()]).T

    #from sklearn.svm import OneClassSVM
    #clf = OneClassSVM(nu=0.261, gamma=0.05)
    #clf.fit(X)
    #Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    #Z = Z.reshape(xx1.shape)
    #CS = plt.contour(x, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
    #                 levels=10)
    #CB = plt.colorbar(CS, shrink=0.8, extend='both')
    #plt.scatter(X['var'], X['zc'], .8)

    #plt.title('Negative log-likelihood predicted by a GMM')
    #plt.axis('tight')
    #plt.show()
    #sys.exit()

    df = pd.read_csv('New_Gaps_output_data_noise_KIC_-1_fullKep.txt')#-1.txt')#80.txt')
    print()
    rg.rename(columns = {'KICID':'KIC'}, inplace = True)

    #print(df)
    #print(df)
    df = df.dropna(axis='rows')
    #df = df.merge(rg, how='inner', on=['KIC'])
    #print(df.head())
    new_X = df[['KIC', 'var', 'zc', 'hoc', 'mc']]
    #new_X = df[df['KIC'].isin(rg['KIC'])][['KIC', 'numax_y', 'var', 'zc', 'hoc', 'mc']]
    new_X['var'] = np.log10(new_X['var'])
    new_X = new_X[['var', 'zc', 'hoc']]
    #new_X = new_X[['zc', 'hoc']]
    #plt.scatter(df[df['']])

    #from sklearn.ensemble import IsolationForest
    #clf = IsolationForest(contamination=0.5)
    #clf.fit(new_X)
    #y_pred_train = clf.predict(new_X)
    #print(len(y_pred_train[y_pred_train > 0]), len(new_X))
    #y_pred_test = clf.predict(new_X)
    #print(y_pred_test)

    #print(len(y_pred_test[y_pred_test > 0]), len(y_pred_test))
    #plt.scatter(X['zc'], X['hoc'], c=y_pred_train, marker='x')
    #plt.scatter(new_X['zc'], new_X['hoc'], c=y_pred_train, marker='.')
    #plt.colorbar()
    #plt.clim(-1, 1)
    #plt.axis('tight')
#    plt.show()
    #sys.exit()

    from scipy.spatial.distance import cdist
    print(np.shape(new_X.iloc[0].reshape(1,-1)))
    print(np.shape(X))
    dists = np.zeros(len(new_X))
    for i in tqdm(range(len(new_X)), total=len(new_X)):
        dists[i] = np.min(cdist(X.values, new_X.iloc[i].values.reshape(1,-1), metric = 'seuclidean'))#'mahalanobis'))
    new_X['dist'] = np.log10(dists)
    #new_X['numax'] = df[df['KIC'].isin(rg['KIC'])]['numax_y']
    plt.scatter(new_X['zc'], new_X['hoc'], c=new_X['dist'], marker='.')
    plt.colorbar()
    plt.axis('tight')
    plt.show()

    plt.hist(new_X['dist'], bins=100, density=True)
    print(len(new_X['dist'][new_X['dist'] < -2.0]))
    plt.show()

    from sklearn.mixture import GaussianMixture as GMM
    # fit models with 1-10 components
    N = np.arange(10, 11)
    models = [None for i in range(len(N))]

    for i in tqdm(range(len(N)), total=len(N)):
        models[i] = GMM(N[i]).fit(new_X['dist'].values.reshape(-1,1))

    # compute the AIC and the BIC
    AIC = [m.aic(new_X['dist'].values.reshape(-1,1)) for m in models]
    BIC = [m.bic(new_X['dist'].values.reshape(-1,1)) for m in models]

    M_best = models[np.argmin(BIC)]
    x = np.linspace(-3, 1, 1000).reshape(-1,1)
    preds = M_best.predict(new_X['dist'].values.reshape(-1,1))
    print(M_best.means_)
    print(preds)
    logprob = M_best.score_samples(x)
    responsibilities = M_best.predict_proba(x)
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    #plt.scatter(new_X['numax'], new_X['dist'], marker='.')
    #plt.show()

    plt.hist(new_X['dist'], bins=100, density=True, histtype='step')
    plt.plot(x, pdf, '-k')
    plt.plot(x, pdf_individual, '--k')
    plt.show()

    for i in range(10):
        print(M_best.means_[i])
        plt.scatter(new_X['zc'][preds == i], new_X['hoc'][preds == i], marker='.')
        plt.show()

    print(np.min(new_X['dist']))
    cutoff = -1.
    print(len(new_X[new_X['dist'] < cutoff]))
    plt.scatter(X['zc'], X['hoc'], marker='.', zorder=2, alpha=0.1)
    plt.scatter(new_X['zc'][new_X['dist'] < cutoff], new_X['hoc'][new_X['dist'] < cutoff], c=new_X['dist'][new_X['dist'] < cutoff], marker='.')
    plt.colorbar()
    plt.axis('tight')
    plt.show()

    df_new = df[new_X['dist'] < cutoff]
    df_new['dist'] = new_X['dist'][new_X['dist'] < cutoff]
    print(df_new.head())
    print("Yu in whole sample: ", sum(df['KIC'].isin(rg['KIC'])))
    print("Number of candidate solar-like oscillators: ", len(df_new))
    plt.scatter(df_new['zc'], df_new['hoc'], c=df_new['dist'], marker='.')
    plt.colorbar()
    plt.scatter(X['zc'][df['KIC'].isin(rg['KIC'])], X['hoc'][df['KIC'].isin(rg['KIC'])], marker='x', zorder=2, color='k', alpha=0.5)
    plt.axis('tight')
    plt.show()

    #plt.scatter(df_new['zc'], df_new['hoc'], c=df_new['dist'], marker='.')
    #plt.colorbar()
    #print(np.logical_not(df['KIC'].isin(rg['KIC'])))
    #print(new_X[df['KIC'].isin(rg['KIC'])])
    plt.scatter(new_X['zc'][df['KIC'].isin(rg['KIC'])], new_X['hoc'][df['KIC'].isin(rg['KIC'])], marker='x', zorder=2, color='k', alpha=0.5)
    plt.axis('tight')
    plt.show()


    poss_rg = df_new['KIC'][df_new['KIC'].isin(rg['KIC'])]
    print("Yu in my sample: ", sum(df_new['KIC'].isin(rg['KIC'])))
    poss_rg = df['KIC'][df['KIC'].isin(rg['KIC'])]
    print(len(poss_rg))
    sys.exit()




    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(max_samples=len(X))
    clf.fit(X)
    y_pred_train = clf.predict(X)
    y_pred_test = clf.predict(new_X)
    print(y_pred_test)

    print(len(y_pred_test[y_pred_test > 0]), len(y_pred_test))
    #plt.scatter(X['zc'], X['hoc'], c=y_pred_train, marker='x')
    plt.scatter(new_X['zc'], new_X['hoc'], c=y_pred_test, marker='.')
    plt.colorbar()
    plt.clim(-1, 1)
    plt.axis('tight')
    plt.show()

    plt.scatter(new_X['zc'][dists < -3], new_X['hoc'][dists < -3], c=y_pred_test[dists < -3], marker='.')
    plt.colorbar()
    plt.clim(-1, 1)
    plt.axis('tight')
    plt.show()

    print(new_Z.ravel())
    plt.scatter(X['zc'], X['hoc'], c=full_Z.ravel(), marker='x')
    plt.scatter(new_X['zc'], new_X['hoc'], c=new_Z.ravel(), marker='.')
    plt.colorbar()
    plt.clim(-10, 0)
    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()
    cut = -8
    plt.scatter(X['zc'][full_Z > cut], X['hoc'][full_Z > cut], c=full_Z.ravel()[full_Z > cut], marker='x')
    plt.scatter(new_X['zc'][new_Z > cut], new_X['hoc'][new_Z > cut], c=new_Z.ravel()[new_Z > cut], marker='.', cmap='Blues_r')
    plt.colorbar()
    #plt.clim(-100, 0)
    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()

    #df[new_Z > cut].to_csv('Updated_output_data_noise_Star_-1.txt', index=False)

    print(df['evo'].value_counts())
    print(df[new_Z > cut]['evo'].value_counts())
    print(len(df), len(df[new_Z > cut]['evo']))
