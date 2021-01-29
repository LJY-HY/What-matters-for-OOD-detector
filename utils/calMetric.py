from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
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