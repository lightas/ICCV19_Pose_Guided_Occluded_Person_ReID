import cv2 as cv 
import numpy as np
import scipy
from PIL import Image
import os
import scipy
import shutil
test_dir = '/home/jiaxu/jiaxu/data/Occluded_Duke/All_Occluded_test/bbox_test'
target_dir ='/home/jiaxu/jiaxu/data/Occluded_Duke/Alpha_Position/test/gallery'
import json
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

for img in os.listdir(test_dir):
    imgname = img.split('.')[0]
    img_=Image.open(os.path.join(test_dir,img))
    img_np=np.array(img_)
    imgh,imgw=img_np.shape[0],img_np.shape[1]
    if not os.path.isfile(os.path.join('/home/jiaxu/jiaxu/new_project/AlphaPose/examples/gallery/sep-json/',imgname+'.json')):
        final_label=[1,1,1,1,1,1]
    else:
        with open(os.path.join('/home/jiaxu/jiaxu/new_project/AlphaPose/examples/gallery/sep-json/',imgname+'.json'),'r') as f:
            print('????')
            a=json.load(f)
            p=a['people']
        p_count=0
        for i in range(len(p)):
            p_points=p[i]
            p_points=p_points['pose_keypoints_2d']
            p_points=np.array(p_points)
            p_points=p_points.reshape(18,3)
            p_points=p_points[p_points[:,2]>0.3]
            count=p_points.shape[0]
            if count>p_count:
                final_point=p_points
                p_count=count
            if final_point.shape[0]<3:
                final_label=[1,1,1,1,1,1]
            else:
                label=[0,0,0,0,0,0]
                for j in range(len(final_point)):
                    w,h = final_point[j][:2]
                    if h>=0.0 and h<(1/6.0)*imgh:
                        label[0]=1
                    elif h>=(1/6.0)*imgh and h<(2/6.0)*imgh:
                        label[1]=1
                    elif h>=(2/6.0)*imgh and h<(3/6.0)*imgh:
                        label[2]=1
                    elif h>=(3/6.0)*imgh and h<(4/6.0)*imgh:
                        label[3]=1
                    elif h>=(4/6.0)*imgh and h<(5/6.0)*imgh:
                        label[4]=1
                    elif h>(5/6.0)*imgh and h<imgh:
                        label[5]=1
                final_label=label
    final_label=[str(i) for i in final_label]
    if not os.path.exists(os.path.join(target_dir,imgname.split('_')[0])):

        os.makedirs(os.path.join(target_dir,imgname.split('_')[0]))
    new_img=img.split('.')[0]+'__'+'_'.join(final_label)+'.jpg'
    shutil.copy(os.path.join(test_dir,img),os.path.join(target_dir,imgname.split('_')[0],new_img))
