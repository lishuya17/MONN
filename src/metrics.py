"""
@author: Anna Cichonska
"""

import numpy as np
import copy
from math import sqrt
from scipy import stats
from sklearn import preprocessing,metrics




def rmse(y,f):
    """
    Task:    To compute root mean squared error (RMSE)

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rmse   RSME
    """

    rmse = sqrt(((y - f)**2).mean(axis=0))

    return rmse




def pearson(y,f):
    """
    Task:    To compute Pearson correlation coefficient

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rp     Pearson correlation coefficient
    """

    rp = np.corrcoef(y, f)[0,1]

    return rp




def spearman(y,f):
    """
    Task:    To compute Spearman's rank correlation coefficient

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rs     Spearman's rank correlation coefficient
    """

    rs = stats.spearmanr(y, f)[0]

    return rs




def ci(y,f):
    """
    Task:    To compute concordance index (CI)

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  ci     CI

    References:
    [1] Tapio Pahikkala, Antti Airola, Sami Pietila, Sushil Shakyawar,
    Agnieszka Szwajda, JingTang and Tero Aittokallio.
    Toward more realistic drug-target interaction predictions.
    Briefings in Bioinformatics, 16, pages 325-337, 2014.
    """

    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]

    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0

    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1

    ci = S/z

    return ci




def f1(y,f):
    """
    Task:    To compute F1 score using the threshold of 7 M
             to binarize pKd's into true class labels.

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  f1     F1 score
    """

    y_binary = copy.deepcopy(y)
    y_binary = preprocessing.binarize(y_binary.values.reshape(1,-1), threshold=7.0, copy=False)[0]
    f_binary = copy.deepcopy(f)
    f_binary = preprocessing.binarize(f_binary.values.reshape(1,-1), threshold=7.0, copy=False)[0]

    f1 = metrics.f1_score(y_binary, f_binary)

    return f1




def average_AUC(y,f):

    """
    Task:    To compute average area under the ROC curves (AUC) given ten
             interaction threshold values from the pKd interval [6 M, 8 M]
             to binarize pKd's into true class labels.

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  avAUC   average AUC

    """

    thr = np.linspace(6,8,10)
    auc = np.empty(np.shape(thr)); auc[:] = np.nan

    for i in range(len(thr)):
        y_binary = copy.deepcopy(y)
        y_binary = preprocessing.binarize(y_binary.values.reshape(1,-1), threshold=thr[i], copy=False)[0]
        fpr, tpr, thresholds = metrics.roc_curve(y_binary, f, pos_label=1)
        auc[i] = metrics.auc(fpr, tpr)

    avAUC = np.mean(auc)

    return avAUC