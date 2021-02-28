from __future__ import print_function
import numpy as np
import numpy as np
from scipy import misc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_curve(path):
    tp,fp = list(),list()
    tnr_at_tpr95 = 0
    known = np.loadtxt(path+'confidence_In.txt')
    novel = np.loadtxt(path+'confidence_Out.txt')
    known.sort(axis=0)
    novel.sort(axis=0)
    start = np.min([known.min(),novel.min()])
    end = np.max([known.max(),novel.max()])
    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp = -np.ones([num_k+num_n+1],dtype=int)
    fp = -np.ones([num_k+num_n+1],dtype=int)
    tp[0],fp[0] = num_k,num_n
    k,n = 0,0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]
    tpr95_pos = np.abs(tp/num_k-0.95).argmin()
    tnr_at_tpr95=1.-fp[tpr95_pos]/num_n
    return tp,fp,tnr_at_tpr95

def metric(path):
    tp,fp,tnr_at_tpr95=get_curve(path)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']

    # TNR
    mtype = 'TNR'
    results[mtype] = tnr_at_tpr95
            
    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)
            
    # DTACC
    mtype = 'DTACC'
    results[mtype] = .5 * (tp/tp[0] + 1.-fp/fp[0]).max()
            
    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
    
    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results

def compute_metric(known, novel):
    stype = ""
    
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    
    known.sort()
    novel.sort()
    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])
    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
    fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
    tp[stype][0], fp[stype][0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[stype][l+1:] = tp[stype][l]
            fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
            break
        elif n == num_n:
            tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
            fp[stype][l+1:] = fp[stype][l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[stype][l+1] = tp[stype][l]
                fp[stype][l+1] = fp[stype][l] - 1
            else:
                k += 1
                tp[stype][l+1] = tp[stype][l] - 1
                fp[stype][l+1] = fp[stype][l]
    tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
    tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    results = dict()
    results[stype] = dict()
    
    # TNR
    mtype = 'TNR'
    results[stype][mtype] = tnr_at_tpr95[stype]
    
    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
    fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
    results[stype][mtype] = -np.trapz(1.-fpr, tpr)
    
    # DTACC
    mtype = 'DTACC'
    results[stype][mtype] = .5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max()
    
    # AUIN
    mtype = 'AUIN'
    denom = tp[stype]+fp[stype]
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
    results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
    
    # AUOUT
    mtype = 'AUOUT'
    denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
    results[stype][mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
    
    return results[stype]