import scipy.io
import torch
import numpy as np
import time
import os
import torch.nn.functional as F
#######################################################################
# Evaluate
def evaluate(qf,qf2,qpl,ql,qc,gf,gf2,gpl,gl,gc):
    qf=qf.cuda()
    gf=gf.cuda()
    gpl=gpl.cuda()
    qpl=qpl.cuda()
    qf2=qf2.cuda()
    gf2=gf2.cuda()
    #######Calculate the distance of pose-guided global features

    query2 = qf2
    
    
    qf2=qf2.expand_as(gf2)

    q2=F.normalize(qf2,p=2,dim=1)
    g2=F.normalize(gf2,p=2,dim=1)
    s2=q2*g2
    s2=s2.sum(1) #calculate the cosine distance 
    s2=(s2+1.)/2 # convert cosine distance range from [-1,1] to [0,1], because occluded part distance is set to 0

    ########Calculate the distance of partial features
    query = qf
    overlap=gpl*qpl
    overlap=overlap.view(-1,gpl.size(1)) #Calculate the shared region part label
    
    qf=qf.expand_as(gf)

    q=F.normalize(qf,p=2,dim=2)
    g=F.normalize(gf,p=2,dim=2)
    s=q*g
    
    s=s.sum(2) #Calculate the consine distance 
    s=(s+1.)/2 # convert cosine distance range from [-1,1] to [0,1]
    s=s*overlap
    s=(s.sum(1)+s2)/(overlap.sum(1)+1)
    s=s.data.cpu()
####################
    ###############
    score=s.numpy()
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp,index


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc
