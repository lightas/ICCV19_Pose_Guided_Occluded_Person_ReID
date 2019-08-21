import os
from shutil import copyfile

dataset_path = './dataset/Occluded_Duke/'
processed_datapath = 'processed_data'
if not os.path.isdir(dataset_path):
    raise ValueError('Occluded_Duke not found. Please change the dataset_path')
list_=os.listdir(dataset_path)
if 'bounding_box_train' not in list_ or 'bounding_box_test' not in list_ or 'query' not in list_:
    raise ValueError('Occluded_Duke not found. Please convert Duke to Occluded_Duke')


save_path = os.path.join(dataset_path,processed_datapath)
if not os.path.isdir(save_path):
    os.makedirs(save_path)
    #-----------------------------------------
    #query
    query_path = os.path.join(dataset_path,'query')
    query_save_path = os.path.join(dataset_path, processed_datapath,'query')
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)
    
    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = os.path.join(query_path, name)
            dst_path = os.path.join(query_save_path, ID[0])
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, os.path.join(dst_path,name))
    
    #-----------------------------------------
    #gallery
    gallery_path = os.path.join(dataset_path, 'bounding_box_test')
    gallery_save_path = os.path.join(dataset_path,processed_datapath ,'gallery')
    if not os.path.isdir(gallery_save_path):
        os.mkdir(gallery_save_path)
    
    for root, dirs, files in os.walk(gallery_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = os.path.join(gallery_path, name)
            dst_path = os.path.join(gallery_save_path, ID[0])
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, os.path.join(dst_path,name))
    
    #---------------------------------------
    #train
    train_path = os.path.join(dataset_path, 'bounding_box_train')
    train_save_path = os.path.join(dataset_path,processed_datapath ,'train')
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
    
    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = os.path.join(train_path, name)
            dst_path = os.path.join(train_save_path,ID[0])
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, os.path.join(dst_path, name))
    
