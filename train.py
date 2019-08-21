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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from model import  PCB,ClassBlock
import json
from shutil import copyfile
from torch.utils.data import Dataset
from utils import prepare
from utils.meters import AverageMeter
import torchvision.transforms.functional as tF
import random
version =  torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='PGFA', type=str, help='output model name')
parser.add_argument('--data_dir',default='./dataset/Occluded_Duke/processed_data',type=str, help='training dir path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--lamb', default=0.2, type=float, help='lambda')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--part_num', default=3, type=int, help='part number')
parser.add_argument('--mask_dir',default='./heatmaps/18heatmap_train',type=str, help='attention heapmap dir path')
parser.add_argument('--hidden_dim', default=256, type=int, help='part number')
parser.add_argument('--result_dir',default='./result',type=str, help='path to save result')
opt = parser.parse_args()
TRAIN_MASK_DIR=opt.mask_dir # Generated attention heatmap by pose landmarks
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)
part_size=opt.part_num
# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
#######
transform_train_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
########Convert person image id to label
def id2label(ids):
    dic={}
    for index,id in enumerate(ids):
        dic[id]=index
    return dic
########Dataset
class BaseDataset(Dataset):
    def __init__(self,train_datapath):
        self.datapath=train_datapath
        self.transform=transform_train_list
        self.ids=sorted(os.listdir(self.datapath))
        self.classnum=len(self.ids)
        self.labeldic=id2label(self.ids)
        self.data=[]
        for pid in sorted(self.ids):
            for img in os.listdir(os.path.join(self.datapath,pid)):
                imgpath=os.path.join(self.datapath,pid,img)
                self.data.append((imgpath,pid))
    def __getitem__(self,index):
        imgpath,pid=self.data[index]
        img=Image.open(imgpath).convert('RGB')
        img=transforms.Resize(size=(384,128),interpolation=3)(img)
        mask_name=imgpath.split('/')[-1].split('.')[0]+'.npy'
        mask=np.load(os.path.join(TRAIN_MASK_DIR,mask_name))
        if random.random()<0.5:
            img=tF.hflip(img)
            mask=np.flip(mask,2)

        img=self.transform(img)
        label=self.labeldic[pid]
        mask=torch.from_numpy(mask.copy())
        mask=mask.float()
        return img,label,mask
    def __len__(self):
        return len(self.data)

######################################################################
# Load Data
# ---------

image_datasets =BaseDataset(os.path.join(data_dir,'train')) 

dataloaders =  torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8) # 8 workers may work faster
              
dataset_sizes =  len(image_datasets) 
class_num = image_datasets.classnum

use_gpu = torch.cuda.is_available()
######################################################################
# Training the model
# ------------------
#

y_loss = {} # loss history
y_loss['train'] = []
y_err = {}
y_err['train'] = []

def train_model(model,global_classifier,PCB_classifier,criterion, optimizer,optimizer2,optimizer3, scheduler,scheduler2 ,scheduler3,num_epochs=25):
    '''
    model: the backbone of our pipeline (ResNet50 without FC layer and Average Pooling layer)
    global_classifier: The classifier of Global Feature Branch
    PCB_classsifier: the classifiers of Partial Feature Branch
    optimizer,optimizer2,optimizer3 work on model, global_classifier, PCB_classifier respectively
    scheduler,scheduler2 ,scheduler3 work on optimizer,optimizer2,optimizer3 respectively.
    '''
    since = time.time()


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        scheduler2.step()
        for i in range(part_size):
            scheduler3[i].step()
        model.train(True)  # Set model to training mode
        global_classifier.train(True)
        for i in range(part_size):
            PCB_classifier[i].train(True)
        
        running_loss = {}
        for i in range(part_size):
            running_loss[i]=AverageMeter()  # Partial Losses
        running_loss2=AverageMeter()  #Global Feature Loss
        # Iterate over data.

        for it,data in enumerate(dataloaders):
            # get the inputs
            inputs, labels,masks = data
            now_batch_size,c,h,w = inputs.shape
            if now_batch_size<opt.batchsize: # skip the last batch
                continue
            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                masks = masks.cuda()
            else:
                inputs, labels,masks = Variable(inputs), Variable(labels)
            
            features = model(inputs) # Extract ResNet50 Feature



            #########Partial Feature Branch
            partial_feature=nn.AdaptiveAvgPool2d((opt.part_num,1))(features) #Horizontally partition
            partial_feature=torch.squeeze(partial_feature)
            loss=[]
            for i in range(part_size):
                output=PCB_classifier[i](partial_feature[:,:,i])
                loss_=criterion(output,labels)
                loss.append(loss_)
            for i in range(part_size):
                running_loss[i].update(loss[i].item(),now_batch_size)
            PCBloss=sum(loss)  
            ##########Pose Guided Global Feature Branch
            pg_global_feature_1=nn.AdaptiveAvgPool2d((1,1))(features)
            pg_global_feature_1=torch.squeeze(pg_global_feature_1)
            pg_global_feature_2=torch.cuda.FloatTensor()
            for i in range(18): #There are 18 pose landmarks per image
                mask=masks[:,i,:,:]
                mask=torch.unsqueeze(mask,1)
                mask = mask.expand_as(features)

                pg_feature_=mask*features  #  element-wise multiplication 
                pg_feature_=nn.AdaptiveAvgPool2d((1,1))(pg_feature_)
                pg_feature_=torch.squeeze(pg_feature_)
                pg_feature_=torch.unsqueeze(pg_feature_,2)
                pg_global_feature_2=torch.cat((pg_global_feature_2,pg_feature_),2)
            pg_global_feature_2=nn.AdaptiveMaxPool1d(1)(pg_global_feature_2)
            pg_global_feature_2=torch.squeeze(pg_global_feature_2)
            pg_global_feature=torch.cat((pg_global_feature_1,pg_global_feature_2),1)
            output2=global_classifier(pg_global_feature)
            loss2=criterion(output2,labels)
            loss=opt.lamb*PCBloss+loss2*(1-opt.lamb) 
            ###########################################
            optimizer.zero_grad()
            optimizer2.zero_grad()
            for i in range(part_size):
                optimizer3[i].zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.step()
            for i in range(part_size):
                optimizer3[i].step()

            running_loss2.update(loss2.item(),now_batch_size)

            if it%50==0:
                print('epoch: [{}][{}/{}] \t'.format(epoch,it+1,len(dataloaders)))
                for i in range(part_size):
                    print('Loss1_{}:{:.3f} ({:.3f}) \t'.format(i+1,running_loss[i].val,running_loss[i].avg))
                print('Loss2:{:.3f} ({:.3f}) \t'.format(running_loss2.val,running_loss2.avg))
                        
        if epoch%10 == 9:
            save_network('net',model, epoch)
            save_network('part',global_classifier,epoch)
            for i in range(part_size):
                save_network('clf'+str(i), PCB_classifier[i],epoch)
    return model


#######################################################################
# Save model
#---------------------------
def save_network(name_,network, epoch_label):
    save_filename = '%s_%s.pth'%(name_, epoch_label)
    save_path = os.path.join(opt.result_dir,name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.

model = PCB(class_num)
print(model)
print('*'*20)
global_classifier=ClassBlock(4096,class_num,True,False,opt.hidden_dim)
PCB_classifier={}
for i in range(part_size):
    PCB_classifier[i]=ClassBlock(2048,class_num,True,False,opt.hidden_dim)

if use_gpu:
    model = model.cuda()
    global_classifier=global_classifier.cuda()
    for i in range(part_size):
        PCB_classifier[i].cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=opt.lr*0.1, weight_decay=opt.weight_decay, momentum=0.9, nesterov=True) 
optimizer2 = optim.SGD(global_classifier.parameters(),lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
optimizer3={}
for i in range(part_size):
    optimizer3[i]=optim.SGD(PCB_classifier[i].parameters(),lr=opt.lr,weight_decay=opt.weight_decay,momentum=0.9,nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=40, gamma=0.1)
scheduler3 ={}
for i in range(part_size):
    scheduler3[i]=lr_scheduler.StepLR(optimizer3[i], step_size=40, gamma=0.1)

######################################################################
# Train and evaluate
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join(opt.result_dir,name)
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model,global_classifier,PCB_classifier, criterion,optimizer,optimizer2, optimizer3,exp_lr_scheduler,scheduler2,scheduler3,
                       num_epochs=60)

