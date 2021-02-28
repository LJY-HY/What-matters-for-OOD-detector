import random
import numpy as np
import torch
import torch.nn.functional as F
from utils import calMetric

def detect(all_test_deviations,all_ood_deviations, verbose=True, normalize=True):
    average_results = {}
    for i in range(1,11):
        random.seed(i)
        
        validation_indices = random.sample(range(len(all_test_deviations)),int(0.1*len(all_test_deviations)))
        test_indices = sorted(list(set(range(len(all_test_deviations)))-set(validation_indices)))

        validation = all_test_deviations[validation_indices]
        test_deviations = all_test_deviations[test_indices]

        t95 = validation.mean(axis=0)+10**-7
        if not normalize:
            t95 = np.ones_like(t95)
        test_deviations = (test_deviations/t95[np.newaxis,:]).sum(axis=1)
        ood_deviations = (all_ood_deviations/t95[np.newaxis,:]).sum(axis=1)
        results = calMetric.compute_metric(-test_deviations,-ood_deviations)
        for m in results:
            average_results[m] = average_results.get(m,0)+results[m]
    
    for m in average_results:
        average_results[m] /= i
    if verbose:
        mtypes = ['TNR', 'DTACC', 'AUROC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*average_results['TNR']), end='')
        print(' {val:6.2f}'.format(val=100.*average_results['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*average_results['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*average_results['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*average_results['AUOUT']), end='')
        print('')
    return average_results

def cpu(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cpu()
    return ob

def cuda(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cuda()
    return ob

class Detector:
    def __init__(self,args):
        self.all_test_deviations = None
        self.mins = {}
        self.maxs = {}
        
        self.classes = range(args.num_classes)
    
    def compute_minmaxs(self,net,data_train,train_preds,POWERS=[10]):
        for PRED in self.classes:
            train_indices = np.where(np.array(train_preds)==PRED)[0]
            train_PRED = torch.squeeze(torch.stack([data_train[i][0] for i in train_indices]),dim=1)
            mins,maxs = net.get_min_max(train_PRED,power=POWERS)
            self.mins[PRED] = cpu(mins)
            self.maxs[PRED] = cpu(maxs)
            torch.cuda.empty_cache()
    
    def compute_test_deviations(self,net,data,test_preds, test_confs, POWERS=[10]):
        all_test_deviations = None
        test_classes = []
        for PRED in self.classes:
            test_indices = np.where(np.array(test_preds)==PRED)[0]
            test_PRED = torch.squeeze(torch.stack([data[i][0] for i in test_indices]),dim=1)
            test_confs_PRED = np.array([test_confs[i] for i in test_indices])
            
            test_classes.extend([PRED]*len(test_indices))
            
            mins = cuda(self.mins[PRED])
            maxs = cuda(self.maxs[PRED])
            test_deviations = net.get_deviations(test_PRED,power=POWERS,mins=mins,maxs=maxs)/test_confs_PRED[:,np.newaxis]
            cpu(mins)
            cpu(maxs)
            if all_test_deviations is None:
                all_test_deviations = test_deviations
            else:
                all_test_deviations = np.concatenate([all_test_deviations,test_deviations],axis=0)
            torch.cuda.empty_cache()
        self.all_test_deviations = all_test_deviations
        self.test_classes = np.array(test_classes)
    
    def compute_ood_deviations(self,net,ood,POWERS=[10]):
        ood_preds = []
        ood_confs = []
        
        for idx in range(0,len(ood),128):
            batch = torch.squeeze(torch.stack([x[0] for x in ood[idx:idx+128]]),dim=1).cuda()
            logits = net(batch)
            confs = F.softmax(logits,dim=1).cpu().detach().numpy()
            preds = np.argmax(confs,axis=1)
            
            ood_confs.extend(np.max(confs,axis=1))
            ood_preds.extend(preds)  
            torch.cuda.empty_cache()
        print("Done")
        
        ood_classes = []
        all_ood_deviations = None
        for PRED in self.classes:
            ood_indices = np.where(np.array(ood_preds)==PRED)[0]
            if len(ood_indices)==0:
                continue
            ood_classes.extend([PRED]*len(ood_indices))
            
            ood_PRED = torch.squeeze(torch.stack([ood[i][0] for i in ood_indices]),dim=1)
            ood_confs_PRED =  np.array([ood_confs[i] for i in ood_indices])
            mins = cuda(self.mins[PRED])
            maxs = cuda(self.maxs[PRED])
            import pdb;pdb.set_trace()
            ood_deviations = net.get_deviations(ood_PRED,power=POWERS,mins=mins,maxs=maxs)/ood_confs_PRED[:,np.newaxis]
            cpu(self.mins[PRED])
            cpu(self.maxs[PRED])            
            if all_ood_deviations is None:
                all_ood_deviations = ood_deviations
            else:
                all_ood_deviations = np.concatenate([all_ood_deviations,ood_deviations],axis=0)
            torch.cuda.empty_cache()
        import pdb;pdb.set_trace()
        self.ood_classes = np.array(ood_classes)
        average_results = detect(self.all_test_deviations,all_ood_deviations)
        return average_results, self.all_test_deviations, all_ood_deviations

def G_p(ob, p):
    temp = ob.detach()
    
    temp = temp**p
    temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
    temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1)))).sum(dim=2) 
    temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],-1)
    
    return temp