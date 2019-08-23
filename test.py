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

name = opt.name
test_dir = opt.test_dir


# set gpu ids
data_dir = test_dir
###
######################################################################
# Load Data
# ---------
data_transforms = transforms.Compose([
        transforms.Resize((384,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


######################################################################
class BaseDataset(Dataset):
    def __init__(self,test_datapath,mask_path):
        self.datapath=test_datapath
        self.transform=data_transforms
        self.mask_path=mask_path
        self.ids = sorted(os.listdir(self.datapath))
        self.classnum=len(self.ids)
        self.data=[]
        for pid in sorted(self.ids):
            for img in os.listdir(os.path.join(self.datapath,pid)):
                imgpath = os.path.join(self.datapath,pid,img)
                cam_id = int(img.split('c')[1][0])
                self.data.append((imgpath,int(pid),int(cam_id),img))
    def __getitem__(self,index):
        imgpath,pid,cam_id,imgname=self.data[index]
        img=Image.open(imgpath).convert('RGB')
        w,h = img.size
        img = self.transform(img)
        mask_name = imgpath.split('/')[-1].split('.')[0]+'.npy'
        mask=np.load(os.path.join(self.mask_path,mask_name))
        mask=torch.from_numpy(mask)
        mask=mask.float()
        return img,pid,cam_id,mask,imgname,h
    def __len__(self):
        return len(self.data)

######################################################################
# Load model
#---------------------------

def load_network(name_,network):
    save_path = os.path.join(opt.result_dir,name,'%s_%s.pth'%(name_,opt.which_epoch))
    network.load_state_dict(torch.load(save_path))
#    pretrained_dict=torch.load(save_path)
#    model_dict=network.state_dict()
#    # 1. filter out unnecessary keys
#    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#    # 2. overwrite entries in the existing state dict
#    model_dict.update(pretrained_dict)
#    network.load_state_dict(model_dict)
    return network

#########################################################################
def extract_global_feature(model,input_):
    output=model(input_)
    return output
###
def extract_partial_feature(model,global_feature,part_num):
    partial_feature=nn.AdaptiveAvgPool2d((part_num,1))(global_feature)
    partial_feature=torch.squeeze(partial_feature,-1)
    partial_feature=partial_feature.permute(0,2,1)
    return partial_feature
###
def extract_pg_global_feature(model,global_feature,masks):
    pg_global_feature_1=nn.AdaptiveAvgPool2d((1,1))(global_feature)
    pg_global_feature_1=pg_global_feature_1.view(-1,2048)
    pg_global_feature_2=[]
    for i in range(18):  
        mask=masks[:,i,:,:]
        mask= torch.unsqueeze(mask,1)
        mask=mask.expand_as(global_feature)
        pg_feature_=mask*global_feature
        pg_feature_=nn.AdaptiveAvgPool2d((1,1))(pg_feature_)
        pg_feature_=pg_feature_.view(-1,2048,1)
        pg_global_feature_2.append(pg_feature_)
    pg_global_feature_2=torch.cat((pg_global_feature_2),2)
    pg_global_feature_2=nn.AdaptiveMaxPool1d(1)(pg_global_feature_2)
    pg_global_feature_2=pg_global_feature_2.view(-1,2048)
    pg_global_feature=torch.cat((pg_global_feature_1,pg_global_feature_2),1)
    pg_global_feature=model(pg_global_feature)
    return pg_global_feature

###
def feature_extractor(data_path,mask_path,pose_path,model,partial_model,global_model):
    total_part_label=[]

    total_partial_feature=[]  #storage of partial feature 
    total_pg_global_feature=[] #storage of pose guided global feature
    total_label=[] #storage of gallery label
    total_cam=[] #storage of gallery cam
    list_img=[] 

    image_dataset = BaseDataset(data_path,mask_path)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=opt.batchsize,
                                                 shuffle=False,num_workers=0)
    for it, data in enumerate(dataloader):
        imgs,pids,cam_ids,masks,imgnames,heights=data
        imgs= imgs.cuda()
        masks=masks.cuda()
        global_feature=extract_global_feature(model,imgs)
        partial_feature = extract_partial_feature(partial_model,global_feature,opt.part_num)
        total_partial_feature.append(partial_feature.data.cpu())
        ##
        pg_global_feature=extract_pg_global_feature(global_model,global_feature,masks)
        total_pg_global_feature.append(pg_global_feature.data.cpu())

        ##
        total_label.extend(pids)
        total_cam.extend(cam_ids)
        list_img.extend(imgnames)
        ###extract part label
        for imgname,h in zip(imgnames,heights):
            imgname=imgname.split('.')[0]
            part_label=part_label_generate(pose_path,imgname,opt.part_num,h.item()) #part label generation
            part_label=torch.from_numpy(part_label)
            part_label=part_label.unsqueeze(0)
            total_part_label.append(part_label.float())
        if it%10==0:
            print('[{}/{}]'.format(it,len(dataloader)))
    total_part_label=torch.cat(total_part_label,0)
    total_partial_feature=torch.cat(total_partial_feature,0)
    total_pg_global_feature=torch.cat(total_pg_global_feature,0)
    total_label = np.array(total_label)
    total_cam=np.array(total_cam)
    return total_part_label,total_partial_feature,total_pg_global_feature,total_label,total_cam,list_img




        




# Load Collected data Trained model
print('-------test-----------')
model_structure = PCB(opt.train_classnum)
model = load_network('net',model_structure)
global_model_structure=ClassBlock(4096,opt.train_classnum,True,False,opt.hidden_dim)
global_model=load_network('global',global_model_structure)
global_model.classifier=nn.Sequential()
partial_model={}
for i in range(opt.part_num):
    part_model_=ClassBlock(2048,opt.train_classnum,True,False,opt.hidden_dim)
    partial_model[i]=load_network('partial'+str(i),part_model_)
    partial_model[i].classifier = nn.Sequential()
    partial_model[i].eval()
    partial_model[i]=partial_model[i].cuda()

# Change to test mode
model.eval()
global_model.eval()
model = model.cuda()
global_model=global_model.cuda()
##########Test data path
gallery_path=os.path.join(opt.test_dir,'gallery')
query_path=os.path.join(opt.test_dir,'query')
gllist=os.listdir(gallery_path)
ap = 0.0
count=0

####################
def main():
    print('extracting gallery feature...')
    tgpl,tgf,tgf2,tgl,tgc,gallery_imgs=feature_extractor(gallery_path,opt.gallery_heatmapdir,opt.gallery_posedir,
                                                            model,partial_model,global_model)
    print('gallery feature finished')    

    print('extracting query feature...')
    tqpl,tqf,tqf2,tql,tqc,query_imgs=feature_extractor(query_path,opt.query_heatmapdir,opt.query_posedir,
                                                            model,partial_model,global_model)
    print('query feature finished')    

    print('CMC calculating...')    

    count=0
    CMC=torch.IntTensor(len(gallery_imgs)).zero_()    
    ap=0.0

    for qf,qf2,qpl,ql,qc in zip(tqf,tqf2,tqpl,tql,tqc):
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

if __name__=='__main__':
    main()
