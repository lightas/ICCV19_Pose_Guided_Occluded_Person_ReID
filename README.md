# ICCV19 Pose-Guided Feature Alignment Occluded Person ReID
This is the pytorch implementation  and dataset of the  ICCV2019 paper *"Pose-Guided Feature Alignment for Occluded Person Re-identification"*
## Preparation

### Dependencies
 - Python 3.7
 - Pytorch 1.0
 - Numpy

## Occluded-DukeMTMC Dataset Preparation

The **Occluded-DukeMTMC** dataset is re-splited based on the DukeMTMC dataset. Since the privacy implications of the data set are being considered, we cannot release the images of Occluded-DukeMTMC. We only release the image name lists of our Occluded-DukeMTMC dataset in './dataset/Occluded_Duke'. If you can access to the DukeMTMC-reid dataset, you can easily convert DukeMTMC-reid to Occluded-DukeMTMC by running the following code 

```
cd dataset
python convert_duke_to_occduke.py /path/to/DukeMTMC-reID.zip
cd ..

```

## Pose landmarks extraction (optional)

Use [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) to extract pose landmarks of the training set and testing set.
```
cd AlphaPose
sh infer.sh
cd ..
```


**Or you can download our extracted pose landmarks and generated heatmaps.**

## Download pose landmarks and heatmaps

Download [pose landmarks](https://drive.google.com/file/d/1taQBm34ZTICINK9gSORj-XbBxknnZpam/view?usp=sharing) and [heatmaps](https://drive.google.com/file/d/1VVHgt-9FmBFaxAJ-70DSWyyK43LfO97l/view?usp=sharing) into the root path. Unzip them.

## Train

```
python train.py
```

## Test
```
python test.py
```


## Inference on Partial_REID or Partial_iLIDS

./dataset2.zip : Partial_REID and Partial_iLIDS with holistic images.

You can directly use the processed data to run the code. 

Data link: https://drive.google.com/file/d/1ErCEQsNHSHpgZF3-NNj6_OH322vpk8gn/view?usp=sharing

Model link:  https://drive.google.com/file/d/1VarCCCaWZlDYX3La2r8VZpB9rZoHocMm/view?usp=sharing

Heatmaps link:  https://drive.google.com/file/d/1VAmMgGym9XfxAMeq_YmzydDI50KTqnsl/view?usp=sharing
```
   GALLERY_DIR='/your/path/to/heatmaps/Partial_REID/18heatmap_gallery'
   QUERY_DIR='/your/path/to/heatmaps/Partial_REID/18heatmap_query'
   gallery_pose_dir='your/path/to/heatmaps/Partial_REID/gallery_json_1'
   query_pose_dir='your/path/to/heatmaps/Partial_REID/query_json_1'
   python test.py --name market_ckp --part_num 6 --test_dir /your/dataset/path/ —-gallery_heatmapdir $GALLERY_DIR --query_heatmapdir $QUERY_DIR --gallery_posedir $gallery_pose_dir --query_posedir $query_pose_dir --train_classnum 751
```


## Citation
Please cite this paper in your publications if it helps your research:
```
@inproceedings{miao2019pose,
  title={Pose-guided feature alignment for occluded person re-identification},
  author={Miao, Jiaxu and Wu, Yu and Liu, Ping and Ding, Yuhang and Yang, Yi},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={542--551},
  year={2019}
}
```
