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


if __name__=="__main__":

    # Read in data wrt kic and fill etc.
    #fill = np.loadtxt('n_data_kics.txt')
    # Read in data + preprocessing
    #data = np.loadtxt('New_output_data_noise_27.txt')
    #new_data = np.c_[fill, data]
    #df = pd.DataFrame(new_data,
    NDAYS = 80#27
    dataset = 'KIC'
    df = pd.read_csv('Updated_output_data_noise_'+str(dataset)+'_'+str(NDAYS)+'.txt')
    print(df.head())
    if dataset == 'KIC':
        df['KIC'] = df['KIC'].astype(int)
        #print(len(df))
        KICS = df['KIC'][df['evo'] == 6].values.astype(int)
        all_kics = pd.read_csv('/home/kuszlewicz/Dropbox/Python/Clumpiness/apokasc_DR01.dat', delimiter=r'\s+')
        idxs = all_kics['KIC'].isin(KICS)
        tmp = all_kics[idxs]
        tmp = tmp[tmp['numax'] > 180]
        tmp['KIC'] = tmp['KIC'].astype(int)

        df['evo'][df['KIC'].isin(tmp['KIC'])] = 0

        bad_kics = np.loadtxt('../Tables/Bad_dwarfs_from_Keaton.txt')
        print(np.shape(df))
        print(df.head())

        df = df[~df['KIC'].isin(bad_kics)]


        print(len(df))

        #df.dropna(inplace=True)
        print(len(df))
        print(np.unique(df['evo'], return_counts=True))

        #if NDAYS == -1:
        #     df = df[(df['ndata'] > (0.5 * 4 * 365.25 * 86400.0) / (29.4 * 60.0))]
        #else:
        #     df = df[(df['ndata'] > (0.5*NDAYS*86400.0) / (29.4*60.0))]
    df = df.sample(frac=1).reset_index(drop=True)

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
    print(np.sum(df['var'].isnull()), np.shape(df))


    plt.scatter(df['var'], df['zc'], c=df['fill'], marker='.')
    plt.colorbar()
    plt.show()
    X = df[['var', 'zc', 'hoc', 'mc']]
    X = X[X['zc'] > 0]
    X = np.log(X[['var', 'zc']])
    print(np.log(X['var']))
    print(np.min(X['var']), np.max(X['var']))
    print(np.min(X['zc']), np.max(X['zc']))

    # display predicted scores by the model as a contour plot
    x = np.linspace(np.min(X['var']) - 0.1*np.min(X['var']), np.max(X['var']) + 0.1*np.max(X['var']), 100)
    y = np.linspace(np.min(X['zc']) - 0.1*np.min(X['zc']), np.max(X['zc']) + 0.1*np.max(X['zc']), 100)
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


    from sklearn import mixture
    from matplotlib.colors import LogNorm
    from tqdm import tqdm as tqdm
#    N = 20
#    bic = np.zeros(N)
#    for i in tqdm(range(1, N+1), total=N):#
#        clf = mixture.GaussianMixture(n_components=i, covariance_type='full')
#        clf.fit(X)
    #    bic[i-1] = clf.bic(X)

    #plt.plot(np.linspace(1, len(bic), len(bic)), bic, 'o')
    #plt.show()
    #n_comps = np.where(bic == np.min(bic))[0][0] + 1
    #print("Optimal number of components: {}".format(n_comps))
    clf = mixture.GaussianMixture(n_components=11, covariance_type='full')
    clf.fit(X)
    # display predicted scores by the model as a contour plot
    x = np.linspace(np.min(X['var']) - 0.1*np.min(X['var']), np.max(X['var']) + 0.1*np.max(X['var']), 100)
    y = np.linspace(np.min(X['zc']) - 0.1*np.min(X['zc']), np.max(X['zc']) + 0.1*np.max(X['zc']), 100)
    x, Y = np.meshgrid(x, y)
    XX = np.array([x.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(x.shape)

    CS = plt.contour(x, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                     levels=10)
    #CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X['var'], X['zc'], .8)

    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()



    NDAYS = -1
    dataset = 'EPIC' #'Star'
    #df = pd.read_csv('Updated_output_data_noise_'+str(dataset)+'_clean_'+str(NDAYS)+'.txt')
    df = pd.read_csv('Updated_output_data_noise_'+str(dataset)+'_'+str(NDAYS)+'.txt')
    print(df)
    print(df['evo'].value_counts())
    df = df.fillna(value=1e-6)
    df[df['evo'] == 1e-6] = 0
    df[df['var'] == 0] = 1e-6
    df[df['zc'] == 0] = 1e-6
    df[df['hoc'] == 0] = 1e-6
    df[df['mc'] == 0] = 1e-6
    print(df)
    new_X = df[['var', 'zc', 'hoc', 'mc']]
    new_X = new_X[new_X['zc'] > 0]
    new_X = np.log(new_X[['var', 'zc']])
    full_Z = clf.score_samples(X)
    new_Z = clf.score_samples(new_X)

    #plt.scatter(df[df['']])

    plt.hist(full_Z.ravel(), bins=100, density=True, histtype='step')
    plt.hist(new_Z.ravel(), bins=100, density=True, histtype='step')
    plt.show()

    print(new_Z.ravel())
    plt.scatter(X['var'], np.exp(X['zc']), c=full_Z.ravel(), marker='x')
    plt.scatter(new_X['var'], np.exp(new_X['zc']), c=new_Z.ravel(), marker='.')
    plt.colorbar()
    plt.clim(-10, 0)
    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()
    cut = -8
    plt.scatter(X['var'][full_Z > cut], np.exp(X['zc'][full_Z > cut]), c=full_Z.ravel()[full_Z > cut], marker='x')
    plt.scatter(new_X['var'][new_Z > cut], np.exp(new_X['zc'][new_Z > cut]), c=new_Z.ravel()[new_Z > cut], marker='.', cmap='Blues_r')
    plt.colorbar()
    #plt.clim(-100, 0)
    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()

    #df[new_Z > cut].to_csv('Updated_output_data_noise_Star_-1.txt', index=False)

    print(df['evo'].value_counts())
    print(df[new_Z > cut]['evo'].value_counts())
    print(len(df), len(df[new_Z > cut]['evo']))
