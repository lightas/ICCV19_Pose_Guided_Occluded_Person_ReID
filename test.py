# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import time
import os
import scipy.io
from model import  PCB, ClassBlock
from PIL import Image
from shared_region_evaluate import evaluate
from utils import prepare
from utils.part_label import part_label_generate
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='2', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='59', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./dataset/Occluded_Duke/processed_data',type=str, help='./test_data')
parser.add_argument('--result_dir',default='./result',type=str, help='result path')
parser.add_argument('--name', default='PGFA', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--gallery_heatmapdir',default='./heatmaps/18heatmap_gallery',type=str, help='gallery heatmap path')
parser.add_argument('--query_heatmapdir',default='./heatmaps/18heatmap_query',type=str, help='query heatmap path')
parser.add_argument('--gallery_posedir',default='./test_pose_storage/gallery/sep-json',type=str, help='gallery pose path')
parser.add_argument('--query_posedir',default='./test_pose_storage/query/sep-json',type=str, help='query pose path')

parser.add_argument('--train_classnum', default=702, type=int, help='train set class number')
parser.add_argument('--part_num', default=3, type=int, help='part_num')
parser.add_argument('--hidden_dim', default=256, type=int, help='hidden_dim')
opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
data_dir = test_dir
###
use_gpu = torch.cuda.is_available()
######################################################################
# Load Data
# ---------
data_transforms = transforms.Compose([
        transforms.Resize((384,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


######################################################################
######################################################################
# Load model
#---------------------------

def load_network(name_,network):
    save_path = os.path.join(opt.result_dir,name,'%s_%s.pth'%(name_,opt.which_epoch))
#    network.load_state_dict(torch.load(save_path))
    pretrained_dict=torch.load(save_path)
    model_dict=network.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network


# Load Collected data Trained model
print('-------test-----------')
model_structure = PCB(opt.train_classnum)
model = load_network('net',model_structure)
part_model_structure=ClassBlock(4096,opt.train_classnum,True,False,opt.hidden_dim)
part_model=load_network('part',part_model_structure)
# Change to test mode
part_model.classifier=nn.Sequential()
model = model.eval()
part_model=part_model.eval()
if use_gpu:
    model = model.cuda()
    part_model=part_model.cuda()
##########Test data path
gallery_path=os.path.join(opt.test_dir,'gallery')
query_path=os.path.join(opt.test_dir,'query')
gllist=os.listdir(gallery_path)
ap = 0.0
count=0
#############heatmap path and pose landmark path
GALLERY_DIR=opt.gallery_heatmapdir
QUERY_DIR=opt.query_heatmapdir
gallery_pose_dir=opt.gallery_posedir
query_pose_dir = opt.query_posedir
####################
print('extract_gallery feature...')
###################Extract gallery feature
total_gallery_part_label=torch.Tensor()
total_gallery_partial_feature=[]  #storage of partial feature 
total_gallery_pg_global_feature=[] #storage of pose guided global feature
total_gallery_label=[] #storage of gallery label
total_gallery_cam=[] #storage of gallery cam
gallery_img=[] 
count_g=0
part_label_count=0
for pid in os.listdir(gallery_path): #pid refers to person id
    for img in os.listdir(os.path.join(gallery_path,pid)):
        ####Extract global feature
        gallery_img.append(img)
        img_=Image.open(os.path.join(gallery_path,pid,img))
        w,h=img_.size
        img_=data_transforms(img_)
        img_=torch.unsqueeze(img_,0)
        input_im=Variable(img_.cuda())
        output=model(input_im)
        global_feature=output.data.cpu()
        origin_feature=global_feature
        ####
        gl=img.split('_')[0] #gl refers to gallery label
        if gl[0:2]=='-1':
            gl=-1
        else:
            gl=int(gl)
        
        gc=int(img.split('c')[1][0]) #gc refers to gallery camera id
        total_gallery_label.append(gl)
        total_gallery_cam.append(gc)
        ####
        imgname=img.split('.')[0]
        part_label=part_label_generate(gallery_pose_dir,imgname,opt.part_num,h) #part label generation
        part_label=torch.from_numpy(part_label)

        ###################################Partial Features
        partial_feature=nn.AdaptiveAvgPool2d((opt.part_num,1))(global_feature)
        partial_feature=torch.squeeze(partial_feature)
        partial_feature=partial_feature.permute(1,0)


        final_partial_features=torch.Tensor()
        count_zero=0
        for i,p in enumerate(part_label):
            if p==0:
                a=torch.zeros(1,2048)
                final_partial_features=torch.cat((final_partial_features,a),0)
                count_zero+=1
            else:
                a = partial_feature[i]
                a = a.view(1,2048)
                final_partial_features=torch.cat((final_partial_features,a),0) 
        part_label=part_label.float()
        part_label = torch.unsqueeze(part_label,0)
        final_partial_features=torch.unsqueeze(final_partial_features,0)
        total_gallery_partial_feature.append(final_partial_features)
        total_gallery_part_label=torch.cat((total_gallery_part_label,part_label),0)
#        #################
#        #######Pose guided global feature
        maskname =img.split('.')[0]+'.npy'
        masks=np.load(os.path.join(GALLERY_DIR,maskname))
        masks=torch.from_numpy(masks)
        masks=masks.float()
        masks=torch.unsqueeze(masks,0)
        pg_global_feature_1=nn.AdaptiveAvgPool2d((1,1))(origin_feature) #Avg Pool Global Feature
        pg_global_feature_1=pg_global_feature_1.view(1,2048)
        pg_global_feature_2=torch.Tensor()
        for i in range(18):                     #masks multiply global feature element-wisely
            mask=masks[:,i,:,:]
            mask=torch.unsqueeze(mask,1)
            mask=mask.expand_as(origin_feature)
            pg_feature_=mask*origin_feature
            pg_feature_=nn.AdaptiveAvgPool2d((1,1))(pg_feature_)
            pg_feature_=pg_feature_.view(1,2048,1)
            pg_global_feature_2=torch.cat((pg_global_feature_2,pg_feature_),2)
        pg_global_feature_2=nn.AdaptiveMaxPool1d(1)(pg_global_feature_2)
        pg_global_feature_2=pg_global_feature_2.view(1,2048) #masked global feature
        pg_global_feature=torch.cat((pg_global_feature_1,pg_global_feature_2),1)
        pg_global_feature=pg_global_feature.cuda()
        pg_global_feature=part_model(pg_global_feature)
        pg_global_feature=pg_global_feature.data.cpu()
        total_gallery_pg_global_feature.append(pg_global_feature)

        count_g+=1
###################################
total_gallery_partial_feature=torch.cat(total_gallery_partial_feature,0)
total_gallery_pg_global_feature=torch.cat(total_gallery_pg_global_feature,0)
total_gallery_label=np.array(total_gallery_label)
total_gallery_cam=np.array(total_gallery_cam)
################################
tgl=total_gallery_label
tgc=total_gallery_cam
##################################################

tgf=total_gallery_partial_feature
tgf2=total_gallery_pg_global_feature
tgpl=total_gallery_part_label
CMC = torch.IntTensor(tgf.size(0)).zero_()
#################################
##################################
print('gallery feature finish')
#################################query feature extration and matching
count_q=0                
for pid in os.listdir(query_path):
    for img in os.listdir(os.path.join(query_path,pid)):
        img_=Image.open(os.path.join(query_path,pid,img))
        w,h=img_.size
            
        img_=data_transforms(img_)
        img_=torch.unsqueeze(img_,0)
        input_im=Variable(img_.cuda())
        output=model(input_im)
        global_feature=output.data.cpu()
        ql=int(img.split('_')[0]) #query label
        qc=int(img.split('c')[1][0])# query camera


        ##############

        imgname=img.split('.')[0]
        part_label=part_label_generate(query_pose_dir,imgname,opt.part_num,h) #query part label
        part_label = torch.from_numpy(part_label)

       ##########Query pose guided global feature
        maskname =img.split('.')[0]+'.npy'
        masks=np.load(os.path.join(QUERY_DIR,maskname))
        masks=torch.from_numpy(masks)
        masks=masks.float()
        masks=torch.unsqueeze(masks,0)
        pg_global_feature_1=nn.AdaptiveAvgPool2d((1,1))(global_feature)
        pg_global_feature_1=pg_global_feature_1.view(1,2048)
        pg_global_feature_2=torch.Tensor()
        for i in range(18):
            mask=masks[:,i,:,:]
            mask=torch.unsqueeze(mask,1)
            mask=mask.expand_as(global_feature)
            pg_feature_=mask*global_feature
            pg_feature_=nn.AdaptiveAvgPool2d((1,1))(pg_feature_)
            pg_feature_=pg_feature_.view(1,2048,1)
            pg_global_feature_2=torch.cat((pg_global_feature_2,pg_feature_),2)
        pg_global_feature_2=nn.AdaptiveMaxPool1d(1)(pg_global_feature_2)
        pg_global_feature_2=pg_global_feature_2.view(1,2048)
        pg_global_feature=torch.cat((pg_global_feature_1,pg_global_feature_2),1)
        pg_global_feature=pg_global_feature.cuda()
        pg_global_feature=part_model(pg_global_feature)
        pg_global_feature=pg_global_feature.data.cpu()
        qf2=pg_global_feature #qf2 refers to query pose-guided global feature
       ###########Query partial feature
        partial_feature=nn.AdaptiveAvgPool2d((opt.part_num,1))(global_feature)
        partial_feature=torch.squeeze(partial_feature)
        partial_feature=partial_feature.permute(1,0)

            


        final_partial_features=torch.Tensor()
        count_zero=0
        for i,p in enumerate(part_label):
            if p==0:
                a=torch.zeros(1,2048)
                final_partial_features=torch.cat((final_partial_features,a),0)
                count_zero+=1
            else:
                a=partial_feature[i]
                a=a.view(1,2048)
                final_partial_features=torch.cat((final_partial_features,a),0) 
        

        qf=final_partial_features ##qf refers to query partial features
        qpl=part_label.float() ## qp refers to query part label
###########################

###########################Evaluation
        ####
        # qf:query partial features; qf2:query pose-guided global features; qpl:query part label; ql:query label; qc:query camera id
        # tgf:total gallery partial features; tgf2:total gallery pose-guided global feature; tgpl:total gallery part label;
        #  tgl:total gallery label; tgc:total gallery camera id
        ###
        (ap_tmp, CMC_tmp),index = evaluate(qf,qf2,qpl,ql,qc,tgf,tgf2,tgpl,tgl,tgc) #
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        count+=1

CMC = CMC.float()
CMC = CMC/count #average CMC
print('Rank@1    Rank@5   Rank@10    mAP')
print('---------------------------------')
print('{:.4f}    {:.4f}    {:.4f}    {:.4f}'.format(CMC[0],CMC[4],CMC[9],ap/count))
