# HEDNet

It is the official code release of [HEDNet](https://arxiv.org/pdf/2310.20234.pdf) on the nuScenes dataset.

### Results on NuScenes
We implemented HEDNet on NuScenes based on mmdetection3d, because the TransFusion-L implemented on OpenPCDet achieved lower accuracy than on mmdetection3d. We will unify the code in the future.

#### Validation set
|Model|   mATE |  mASE  |  mAOE  | mAVE  | mAAE  |  mAP  |  NDS   |                                              download                                              |
|----------------------------------------------------------------------------------------------------|-------:|:------:|:------:|:-----:|:-----:|:-----:|:------:|:--------------------------------------------------------------------------------------------------:|
| [HEDNet](https://github.com/zhanggang001/HEDNet-nusc/blob/master/configs/hednet/hednet_transfusion_L_nusc.py)                         | 27.5 | 25.1 | 26.3 |	23.3 | 18.7 | 67.0 | 71.4 | [ckpt](https://cloud.tsinghua.edu.cn/f/40f6d51e038f4c158616/?dl=1) |

#### Test set
|Model| mATE | mASE | mAOE | mAVE | mAAE | mAP | NDS | download |
|---|-------:|:------:|:------:|:-----:|:-----:|:-----:|:------:|:----:|
| [HEDNet](https://github.com/zhanggang001/HEDNet-nusc/blob/master/configs/hednet/hednet_transfusion_L_nusc_trainval.py) | 25.0 | 23.8 | 31.7 | 24.0 | 13.0 | 67.5 | 72.0 | [json](https://cloud.tsinghua.edu.cn/f/bf54afa8d28c4d74affe/?dl=1) |

## Installation and usage

Please refer to [getting_started](https://github.com/open-mmlab/mmdetection3d/blob/v0.18.1/docs/en/getting_started.md) for installation, [usage](https://github.com/open-mmlab/mmdetection3d/blob/v0.18.1/docs/en/1_exist_data_model.md) for usage. We used python 3.8, pytorch 1.10, cuda11.3, spconv-cu113 2.3.3, mmdet3d 0.18.1, mmdet 2.11.0, and mmcv 1.3.13.


## Citation
```
@inproceedings{
  zhang2023hednet,
  title={{HEDN}et: A Hierarchical Encoder-Decoder Network for 3D Object Detection in Point Clouds},
  author={Gang Zhang and Chen Junnan and Guohuan Gao and Jianmin Li and Xiaolin Hu},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
}
```

## Acknowleadgement
This work was supported in part by the National Key Research and Development Program of China (No. 2021ZD0200301) and the National Natural Science Foundation of China (Nos. U19B2034, 61836014) and THU-Bosch JCML center.
