import os
import shutil
import sys
from zipfile import ZipFile


target_dir = './Occluded_Duke/'
origin_duke_dir = os.path.join(target_dir,'DukeMTMC-reID') # the temp folder to save the extracted images


def read_origin_duke_zip():
    zip_file = sys.argv[1] # path to the origin DukeMTMC-reID.zip 
    if not os.path.isfile(zip_file) or zip_file.split('/')[-1] != 'DukeMTMC-reID.zip':
        raise ValueError('Wrong zip file. Please provide correct zip file path')
    print("Extracting zip file")
    with ZipFile(zip_file) as z:
        z.extractall(path=target_dir)
        print("Extracting zip file done")


def makedir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_new_split(split, folder_name):
    # read the re-splited name lists
    with open(os.path.join(target_dir,'{}.list'.format(split)),'r') as f:
        imgs=f.readlines()

    source_split = os.path.join(origin_duke_dir, folder_name)
    target_split = os.path.join(target_dir, folder_name)
    if not os.path.exists(target_split):
        os.makedirs(target_split)

    for img in imgs:
        img = img[:-1]
        target_path = os.path.join(target_split, img)        
        if os.path.isfile(os.path.join(source_split, img)):         
            # If the image is kept in its origin split 
            source_img_path = os.path.join(source_split, img)
        else:
            # If the image is moved to another split
            # We move some occluded images from the gallery split to the new query split
            source_img_path = os.path.join(origin_duke_dir, 'bounding_box_test', img) 
        shutil.copy(source_img_path, target_path)
    print("Generate {} split finished.".format(folder_name))


def main():
    # extract the origin DukeMTMC-reID zip, and save images to a temp folder
    read_origin_duke_zip()

    # generate the new dataset
    generate_new_split(split="train", folder_name='bounding_box_train')
    generate_new_split(split="gallery", folder_name='bounding_box_test')
    generate_new_split(split="query", folder_name='query')

    # remove the temp folder
    shutil.rmtree(origin_duke_dir)

    print("\nSuccessfully generate the new Occluded-DukeMTMC dataset.")
    print("The dataset folder is {}".format(target_dir))


if __name__ == '__main__':
    main()

