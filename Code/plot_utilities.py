#!/usr/bin/env/ python3

import itertools
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from operator import itemgetter
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, confusion_matrix, matthews_corrcoef
from xgboost import XGBClassifier



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          #cmap=plt.cm.Blues):
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = plt.text(j, i, cm[i, j].round(3), fontsize=14,
               	  horizontalalignment="center",
               	  color="black")#"white" if cm[i, j] > thresh else "black")
        #if cm[i,j] > thresh:
        #    text.set_path_effects([path_effects.Stroke(linewidth=0.1, foreground='black'),
        #                            path_effects.Normal()])
    plt.colorbar(label=r'Fraction of labels correctly predicted')
    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
