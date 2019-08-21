# The Occluded-DukeMTMC Dataset

This is the **Occluded-DukeMTMC** dataset from the ICCV2019 paper *"Pose-Guided Feature Alignment for Occluded Person Re-Identification"*

## Dataset Discription

The **Occluded-DukeMTMC** dataset is designed for the *occluded person re-id problem*. We re-splited the DukeMTMC-reID dataset to generate the new Occluded-DukeMTMC dataset. Different from the original one, **all query images** and 10% gallery images in the new dataset are occluded person images. Therefore, there always exists at least one occluded image in calculating pairwise distance between query and gallery images. More details of the dataset can be found in our paper [[link]()].

## Dataset Preparation

Since the privacy implications of the DukeMTMC dataset are being considered, we cannot release the images of Occluded-DukeMTMC. We only provide the image name lists of our Occluded-DukeMTMC dataset in './Occluded_Duke'. You can easily convert DukeMTMC-reid to Occluded-DukeMTMC by running the following script：

```
python convert_duke_to_occduke.py /path/to/DukeMTMC-reID.zip 
```

The input is the origin **zip** file of the DukeMTMC-reID dataset. 
The script will generate the new Occluded-DukeMTMC dataset in the folder ***Occluded_Duke***, contains sub-folders ***bounding\_box\_train***, ***bounding\_box\_test*** and ***query***, which has the same structure as the original one. So previous codes that run on DukeMTMC-reid can be directly applied on the new Occluded-DukeMTMC dataset.

Please cite the following two papers if this dataset helps your research.

```
@inproceedings{miao2019PGFA,
  title={Pose-Guided Feature Alignment for Occluded Person Re-Identification},
  author={Miao, Jiaxu and Wu, Yu and Liu, Ping and Ding, Yuhang and Yang, Yi},
  booktitle={ICCV},
  year={2019}
}
@inproceedings{ristani2016MTMC,
  title = {Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking},
  author = {Ristani, Ergys and Solera, Francesco and Zou, Roger and Cucchiara, Rita and Tomasi, Carlo},
  booktitle = {European Conference on Computer Vision workshop on Benchmarking Multi-Target Tracking},
  year = {2016}
}
```
