#!/usr/bin/env/ python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sys
import xgboost as xgb

from scipy import interp
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from utilities import *
from plot_utilities import *
from sklearn.preprocessing import label_binarize

# Set seed to ensure reproducible results!
np.random.seed(0)
random.seed(0)

def find_nearest(array, value):
    #https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class Train:

    def __init__(self, df, dataset, NDAYS):
        self.df = df
        self.dataset = dataset
        self.NDAYS = NDAYS
        self.class_names = ['LLRGB', 'HLRGB', 'ConfRGB', 'CHeB', 'Noise']

    def _days_to_pts(self, x, cadence=29.4/60/24):
        """
        Convert number of days to number of data points at a given 
        cadence.
        """
        return x / cadence

    def _data_cond(self, x, NDAYS):
        """
        Make sure that all of the data has a large enough fill and contains enough data points
        in order to be trained.
        """
        if NDAYS == -1:
            predict_df = x.loc[(x['ndata'] < self._days_to_pts(180)) | (x['evo'] == 6),]
            predict_df = predict_df.loc[predict_df['fill'] > 0.5, ]
            x = x.loc[(x['ndata'] > self._days_to_pts(180)) & (x['evo'] != 6) & (x['fill'] > 0.5), ]
        elif NDAYS == 27:
            predict_df = x.loc[x['evo'] == 6,]
            x = x.loc[(x['evo'] != 6) & (x['fill'] > 0.5), ]
        else:
            predict_df = x.loc[(x['ndata'] < self._days_to_pts(NDAYS/2)) | (x['evo'] == 6),]
            x = x.loc[(x['ndata'] > self._days_to_pts(NDAYS/2)) & (x['evo'] != 6) & (x['fill'] > 0.5), ]
        return x, predict_df

    def _separate_data(self):
        """
        Only train on stars that have enough data for respective classifier
        """
        df, predict_df = self._data_cond(self.df, self.NDAYS)
        return df, predict_df

    def preprocess_Kepler(self):
        """
        Specific preprocessing if training on Kepler data
        """
        self.df['KIC'] = self.df['KIC'].astype(int)

        KICS = self.df.loc[self.df['evo'] == 6, 'KIC'].values.astype(int)

        # Any star with numax > 180 is definitely going to be RGB
        self.df.loc[self.df['numax'] > 180, 'evo'] = 0

        # Bad stars from Keaton
        # This is specific to the training set and should not be here!
        bad_kics = np.loadtxt('/mnt/c/Users/kuszi/Documents/Clumpiness-refactor/Tables/Bad_dwarfs_from_Keaton.txt')
        self.df = self.df[~self.df['KIC'].isin(bad_kics)]

        return self.df

    def preprocess_data(self):
        """
        Preprocess data
        """    
        #print(df['running_var'])
        if self.NDAYS == -1:
            self.df['running_var'] = np.log10(np.abs(self.df['running_var']))
            self.df = self.df.loc[self.df['running_var'] < 2.5, ]

        if self.dataset == 'KIC':
            self.df = self.preprocess_Kepler()

        # Shuffle data
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        # Separate data
        self.df, self.predict_df = self._separate_data()
      
    def setup_for_training(self, features=None, test_size=0.3):
        """
        Set up the features for training
        """
        # Only train classifier on best features
        if features:
            self.X_predict = self.predict_df[features]
        else:
            features = ['KIC', 'numax', 'var', 'zc', 'hoc', 'abs_k_mag', 'mc']

            self.X_predict = self.predict_df[features]
 
        self.X = self.df[features]
	
        # Labels
        self.y = self.df['evo'].astype(int)
        if self.NDAYS != -1:
            # Find unique KIC values            
            kics = self.df.drop_duplicates('KIC')
            # Create train and test sets based of the unique kic numbers
            train_kics, test_kics, train_y, test_y = train_test_split(kics['KIC'].values, 
                                                                      kics['evo'].values,
                                                                      test_size=test_size, 
                                                                      stratify=kics['evo'].values)
            self.X_train = self.X.loc[self.X['KIC'].isin(train_kics),]
            self.X_test = self.X.loc[self.X['KIC'].isin(test_kics),]
            self.Y_train = self.df.loc[self.df['KIC'].isin(train_kics),'evo'].astype(int)
            self.Y_test = self.df.loc[self.df['KIC'].isin(test_kics),'evo'].astype(int)

        else:
            # Train, test split
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,
                                                                                    self.y,
                                                                                    test_size=0.3,
                                                                                    stratify=self.y)
        # Create Dmatrix for use with xgboost
        #print(list(self.X_train))
        self.xg_train = xgb.DMatrix(self.X_train.drop(['KIC', 'numax'], axis=1), label=self.Y_train)
        self.xg_test =  xgb.DMatrix(self.X_test.drop(['KIC', 'numax'], axis=1), label=self.Y_test)

    def train_model(self, xgb_params, num_round=1000, plot=True, save_model=True):
        """
        Train the model
        """

        # Set watchlist and number of rounds
        watchlist = [ (self.xg_train,'train'), (self.xg_test, 'test') ]

        # Progress dictionary
        progress = {}
        # Cross validation - stratified with 10 folds, preserves class proportions
        cvresult = xgb.cv(xgb_params, self.xg_train, num_round, stratified=True, nfold=10, seed = 0,
                          callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
        print(cvresult)
        if plot:
            nrounds = np.linspace(1, num_round+1, num_round)
            # Plot results of simple cross-validation
            plt.errorbar(nrounds, cvresult['train-mlogloss-mean'], yerr=cvresult['train-mlogloss-std'], fmt='o', color='b', label='Training error')
            plt.errorbar(nrounds, cvresult['test-mlogloss-mean'], yerr=cvresult['test-mlogloss-std'], fmt='o', color='r', label='Test error')
            plt.show()

        # Number of rounds used that minimises test error in cross-validation used for last training of model
        self.bst = xgb.train(xgb_params, self.xg_train, 
                        int(cvresult[cvresult['test-mlogloss-mean'] == np.min(cvresult['test-mlogloss-mean'])].index[0]), 
                        watchlist, early_stopping_rounds=100, evals_result=progress)

        print("BEST ITER: ", self.bst.best_ntree_limit) 
        if save_model == True:   
            # Since early-stopping used, save out best iteration
            np.savetxt('best_iter_'+str(self.NDAYS)+'.txt', np.array([self.bst.best_ntree_limit]))
            print('SAVED MODEL!')
            self.bst.save_model(str(self.NDAYS)+'.model')

    def confusion_matrix(self, Y, y_pred, plot=True, normalize=True, class_names=None):
        """
        Compute confusion matrix
        """
        cnf_matrix = confusion_matrix(Y, y_pred.astype(int))
        if class_names is None:
            class_names = ['LLRGB', 'HLRGB', 'ConfRGB', 'CHeB', 'Noise']
        print(class_names)
        print(np.unique(Y, return_counts=True), np.unique(y_pred, return_counts=True))
        print(np.shape(Y), np.shape(y_pred), np.shape(cnf_matrix))
        np.set_printoptions(precision=3)
        if plot:
            plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=normalize,
                              title='')
        return cnf_matrix

    def accuracy(self, Y, y_pred, full=False):
        """
        Return accuracy of classifier
        """
        if full:
            Y_full = self._create_full_class(Y)
            y_pred_full = self._create_full_class(y_pred)
            return accuracy_score(Y_full, y_pred_full)
        else:
            return accuracy_score(Y, y_pred)

    def compute_AUC(self, Y_test, yprob, class_names=None):
        """
        Compute AUC-ROC
        """
#        y = np.bincount(Y_test.values.ravel()[misc])
#        ii = np.nonzero(y)[0]

        if class_names is None:
            class_names = ['LLRGB', 'HLRGB', 'ConfRGB', 'CHeB', 'Noise']
        colours = ['C'+str(i) for i in range(len(class_names))]
        assert len(class_names) == np.shape(yprob)[1]
        # Binarize Labels
        y = label_binarize(Y_test, classes=np.linspace(0, len(class_names)-1, len(class_names)).astype(int))
        # Compute auc/roc etc.
        fpr = dict()
        tpr = dict()
        thresholds = dict()
        roc_auc = dict()
        n_classes = len(class_names)
        for i in range(n_classes):
            fpr[i], tpr[i], thresholds[i] = roc_curve(y[:,i], yprob[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            cond = find_nearest(thresholds[i], 1/len(class_names))
            plt.axvline(fpr[i][cond], color=colours[i], linestyle='--')

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2,
            label=r'ROC curve of class {} (area = {:0.3f})'.format(class_names[i], roc_auc[i]))
        plt.plot([0,1], [0,1], 'k--', lw=2)
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        plt.xlabel(r'False Positive Rate', fontsize=18)
        plt.ylabel(r'True Positive Rate', fontsize=18)
        plt.legend(loc='lower right')

    def predict(self, X_predict, drop=True):
        if 'KIC' and 'numax' in X_predict.columns:
            xg_predict = xgb.DMatrix(X_predict.drop(['KIC', 'numax'], axis=1))
        elif 'KIC' in X_predict.columns:
            xg_predict = xgb.DMatrix(X_predict.drop(['KIC'], axis=1))
        elif 'numax' in X_predict.columns:
            xg_predict = xgb.DMatrix(X_predict.drop(['numax'], axis=1))
        else:
           xg_predict = xgb.DMatrix(X_predict)
        return self.bst.predict(xg_predict, ntree_limit=self.bst.best_ntree_limit)

    @staticmethod
    def _create_full_class(labels):
        new_labels = labels.copy()
        new_labels[(new_labels == 0)|(new_labels == 1)|(new_labels == 2)] = 0
        new_labels[(new_labels == 3)] = 1
        new_labels[(new_labels == 4)] = 2
        return new_labels

if __name__=="__main__":

    # Read in data wrt kic and fill etc.
    NDAYS = 27
    dataset = 'KIC'
    accounting_for_TESS = False #True
#    df = pd.read_csv('Colours_New_Gaps_output_data_noise_KIC_27_APOKASC.csv')
    #df = pd.read_csv('New_Gaps_output_data_noise_'+str(dataset)+'_'+str(NDAYS)+'_APOKASC.csv')
    df = pd.read_csv('Colours_New_Gaps_output_data_noise_'+str(dataset)+'_'+str(NDAYS)+'_APOKASC.csv')

    #df = pd.read_csv('New_Gaps_MAD_MAD_output_data_noise_'+str(dataset)+'_'+str(NDAYS)+'_APOKASC.csv')
    bad_kics = [10403036, 9846355, 5004660, 4936438, 6450613, 10597648, 5395942]
    df = df[~df.KIC.isin(bad_kics)] 

    # Set up training class
    CLP = Train(df, dataset, NDAYS)
    # Preprocess
    CLP.preprocess_data()
    # Set up data for training
    CLP.setup_for_training()
    # Train model
    # setup parameters for xgboost

    # All parameters stored for each Clumpiness model - not ideal but works
    param_full = {'objective': 'multi:softprob',
                  'eval_metric': 'mlogloss',
                  'eta': 0.1,
                    'gamma': 7.5,
                    'max_depth': 3,
                    'min_child_weight': 4,
                    'subsample': 0.7,
                    'colsample': 0.7,
                    'silent': 1,
                    'nthread': 4,
                    'num_class': len(np.unique(CLP.y)),
                    'seed': 0}
    param_180 = {'objective': 'multi:softprob',
                  'eval_metric': 'mlogloss',
                  'eta': 0.375,
                    'gamma': 7.5,
                    'max_depth': 6,
                    'min_child_weight': 1.0,
                    'subsample': 0.5,
                    'colsample': 0.5,
                    'silent': 1,
                    'nthread': 4,
                    'num_class': len(np.unique(CLP.y)),
                    'seed': 0}

    param_80 = {'objective': 'multi:softprob',
                  'eval_metric': 'mlogloss',
                  'eta': 0.275,
                    'gamma': 7.5,
                    'max_depth': 10,
                    'min_child_weight': 2.0,
                    'subsample': 0.85,
                    'colsample': 0.85,
                    'silent': 1,
                    'nthread': 4,
                    'num_class': len(np.unique(CLP.y)),
                    'seed': 0}

    param_27 = {'objective': 'multi:softprob',
                  'eval_metric': 'mlogloss',
                  'eta': 0.025,
                    'gamma': 7.5,
                    'max_depth': 8,
                    'min_child_weight': 3.0,
                    'subsample': 0.5,
                    'colsample': 0.85,
                    'silent': 1,
                    'nthread': 4,
                    'num_class': len(np.unique(CLP.y)),
                    'seed': 0}
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softprob'
    param['eval_metric'] = 'mlogloss'
    # scale weight of positive examples
    param['eta'] = 0.1 #0.275 #0.375# 0.025 #0.1
    param['gamma'] = 7.5
    param['max_depth'] = 3 #10 #6# 8 #3
    param['min_child_weight'] = 4 #2.0 #1.0# 3.0#4
    param['subsample'] = 0.7# 0.85 #0.5# 0.5#0.7
    param['colsample'] = 0.7#0.85 #0.5# 0.85#0.7
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = len(np.unique(CLP.y))
    param['seed'] = 0
    if NDAYS == -1:
        param = param_full
    elif NDAYS == 180:
        param = param_180
    elif NDAYS == 80:
        param = param_80
    elif NDAYS == 27:
        param = param_27
    else:
        sys.exit()

    # Train model
    CLP.train_model(param, num_round=1000, plot=False, save_model=False)

    # Predict on test set
    y_prob = CLP.predict(CLP.X_test, drop=True)
    ylabel = np.argmax(y_prob, axis=1)

    # Predict on training set
    y_prob_train = CLP.predict(CLP.X_train, drop=True)#.drop(['KIC', 'numax'], axis=1))
    ylabel_train = np.argmax(y_prob_train, axis=1)

    # Confusion matrix
    plt.close()
    print ('predicting, classification accuracy = {}'.format(CLP.accuracy(CLP.Y_test, ylabel)))
    # Compute accuracy
    print("Test set full accuracy: {}".format(CLP.accuracy(CLP.Y_test, ylabel, full=True)))
    
    
    #cnf = CLP.confusion_matrix(CLP.Y_test, ylabel, plot=True, normalize=True)
    #plt.savefig('Confusion_matrix_'+str(NDAYS)+'.pdf', bbox_inches="tight")
    #plt.close()


    # Combine the separate RGB classes into one.
    y_prob_new = np.c_[np.sum(y_prob[:,:3], axis=1), y_prob[:,3:]]
    y_prob_new /= np.sum(y_prob_new, axis=1).reshape(-1,1)


    comb_Y_test = CLP.Y_test.values.copy()
    comb_Y_test[(comb_Y_test == 0) | (comb_Y_test == 1) | (comb_Y_test == 2)] = 0
    comb_Y_test[comb_Y_test == 3] = 1
    comb_Y_test[comb_Y_test == 4] = 2
    comb_y_label = np.argmax(y_prob_new, axis=1)
    print(np.shape(comb_Y_test), np.shape(comb_y_label), np.shape(y_prob_new))
    cnf = CLP.confusion_matrix(comb_Y_test, comb_y_label, plot=True, normalize=True, class_names=['RGB', 'CHeB', 'Noise'])
    plt.savefig('Combined_Confusion_matrix_'+str(NDAYS)+'.pdf', bbox_inches="tight")
    plt.close()
    print('Confusion matrix')
    print(cnf)
    
    # Plot of numax vs class
    yp = np.max(y_prob_new, axis=1)
    numax = CLP.X_test['numax'].values
    yp = yp[np.isfinite(numax)]
    numax = numax[np.isfinite(numax)]

    plt.scatter(numax, yp, s=1, color='k')
    #plt.scatter(CLP.X_test['numax'], np.max(y_prob_new, axis=1), s=1, color='k')
    plt.xlabel(r'$\nu_{\mathrm{max}}$ ($\mu$Hz)', fontsize=18)
    plt.ylabel(r'Class probability', fontsize=18)
    plt.xlim(0, 220)
    plt.ylim(0, 1.05)
    plt.savefig('numax_vs_prob_'+str(NDAYS)+'.pdf')
    plt.close()
    print ('predicting, classification accuracy = {}'.format(CLP.accuracy(CLP.Y_test, ylabel)))
#    cond = (CLP.X_test['numax'] < 45) & (CLP.X_test['numax > 35'])
#    print('In tricky region: {}'.format(CLP.accuracy(CLP.Y_test[cond], ylabel[cond])))
    # Compute accuracy
    print("Training set full accuracy: {}".format(CLP.accuracy(CLP.Y_train, ylabel_train, full=True)))
    print("Test set full accuracy: {}".format(CLP.accuracy(CLP.Y_test, ylabel, full=True)))

    # Write out predictions test set
    #np.savetxt(np.c_[CLP.Y_test, ylabel])

    # Number of misclassified
    misc = np.array([ylabel[i] != CLP.Y_test.values[i] for i in range(len(CLP.Y_test))]).ravel()
    print("Number of misclassified: {0} out of {1}".format(sum(int(ylabel[i]) != CLP.Y_test.values[i] for i in range(len(CLP.Y_test))), len(CLP.Y_test)))
  
 
    # AUC-ROC
    CLP.compute_AUC(CLP.Y_test, y_prob)
    plt.savefig('AUC_'+str(NDAYS)+'.pdf')
    plt.close()
    #plt.show()
 
    # Combined AUC-ROC
    CLP.compute_AUC(comb_Y_test, y_prob_new, class_names=['RGB', 'CHeB', 'Noise'])
    plt.savefig('Combined_AUC_'+str(NDAYS)+'.pdf')
    plt.close()
    sys.exit()   

    # Misclassification
    print(np.shape( CLP.X_test), np.shape( CLP.Y_test))
    df = CLP.X_test.copy()
    df['evo'] = CLP. Y_test.values
    df['classification'] = [int(int(ylabel[i]) !=  CLP.Y_test.values[i]) for i in range(len( CLP.Y_test))]
    df.to_csv('misclass_'+str(NDAYS)+'.csv', index=False)
    sys.exit()

    # Create output
    X_train['full_state'] = ytrain_label.astype(int)
    X_train['Flag'] = 0

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
