import cv2 as cv 
import numpy as np
import scipy
from PIL import Image
import os
import scipy
import shutil
from heatmap import CenterGaussianHeatMap
import sys
test_dir = sys.argv[1]
target_dir = sys.argv[2]
import json
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

for img in os.listdir(test_dir):
    imgname = img.split('.')[0]
    img_=Image.open(os.path.join(test_dir,img))
    img_np=np.array(img_)
    imgh,imgw=img_np.shape[0],img_np.shape[1]
    if not os.path.isfile(os.path.join(sys.argv[3],'sep-json',imgname+'.json')):
        final_hmap=np.ones((18,24,8))
    else:
        with open(os.path.join(sys.argv[3],'sep-json',imgname+'.json'),'r') as f:
            a=json.load(f)
            p=a['people']
        p_count=0
        for i in range(len(p)):
            p_points=p[i]
            p_points=p_points['pose_keypoints_2d']
            p_points=np.array(p_points)
            p_points=p_points.reshape(18,3)
            pp=p_points
            pp[pp[:,2]<=0.2]=0.
            p_points=p_points[p_points[:,2]>0.2]
            count=p_points.shape[0]
            if count>p_count:
                final_point=p_points
                final_p=pp
                p_count=count

        
        heatmap=np.zeros((18,imgh,imgw))
        for j in range(final_p.shape[0]):
            if final_p[j].all()==0.:
            #    print('!')
                heatmap[j]=np.zeros((imgh,imgw))
            else:
                w,h = final_p[j][:2]
                w,h=int(w),int(h)
                heatmap[j]=CenterGaussianHeatMap(imgh,imgw,w,h,(imgh*imgw/1000.))
        final_hmap=np.zeros((18,24,8))
        for j in range(18):
            b=Image.fromarray(heatmap[j]*255).convert('L')
            b=b.resize((8,24),Image.BILINEAR)
            b=np.array(b)
            final_hmap[j]=b/255.
           
    if not os.path.exists(target_dir):

        os.makedirs(target_dir)
    np.save(os.path.join(target_dir,imgname+'.npy'),final_hmap)
