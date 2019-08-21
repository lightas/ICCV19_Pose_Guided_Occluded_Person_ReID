# ICCV19 Pose-Guided Feature Alignment Occluded Person ReID
This is the pytorch implementation  and dataset of the  ICCV2019 paper *"Pose-Guided Feature Alignment for Occluded Person Re-identification"*
## Preparation

### Dependencies
 - Python 3.7
 - Pytorch 1.0
 - Numpy

##Occluded-DukeMTMC Dataset 

The **Occluded-DukeMTMC** dataset is re-splited based on the DukeMTMC dataset [1]. Since the privacy implications of the data set are being considered, we cannot release the images of Occluded-DukeMTMC. We only release the image name lists of our Occluded-DukeMTMC dataset in './dataset/Occluded_Duke'. If you can access to the DukeMTMC-reid dataset, you can easily convert DukeMTMC-reid to Occluded-DukeMTMC by running the following code 

```
python ./dataset/convert_duke_to_occduke.py /path/to/DukeMTMC-reID.zip ./dataset/Occluded_Duke/

```

The output is a folder named *occ_duke*, contains sub-folders *bounding\_box\_train*, *bounding\_box\_test* and *query*.


## Dataset Perparation

Preparation: Put the images with the same id in one folder. You may use

```
python prepare.py
```
Remember to change the dataset path to your own path of Occluded-DukeMTMC.

## Pose landmarks extraction (optional)

Use [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) to extract pose landmarks of the training set and testing set.
```
 python3 demo.py --indir /path/to/dataset --outdir /path/of/output --format open --vis_fast
```

Then use the following code to generate heatmaps.
```
python ./heatmaps/new_heatmap.py
```
Remember to change the dataset path to your own path.

**Or you can download our extracted pose landmarks and generated heatmaps.**

##Download pose landmarks and heatmaps

Download [pose landmarks](https://drive.google.com/file/d/1taQBm34ZTICINK9gSORj-XbBxknnZpam/view?usp=sharing) and [heatmaps](https://drive.google.com/file/d/1T55MSPmImCrE-VVLeZPBVG6eR_VD0NPu/view?usp=sharing) into the root path. Unzip them.

## Train

```
sh train_occduke.sh
```

## Test
```
sh test_occduke.sh
```

