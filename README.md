# CASIA_FACE_V5_verification

Verification with ArcFace pretrained model: https://github.com/deepinsight/insightface

- face_det.py: using MTCNN to extract faces and align

- extract_feature.py: extract 512 (or 128)-dim vector of faces

- verification.py: find the best threshold

Put pretrained models follow below structure:

- models\mobilefacenet\json and params file

- models\resnet34

- models\resnet50

- models\resnet100


Download CASIA_FACE_V5

mkdir aligned_faces, faces_pose, features, scores directory

## Citation

CASIA FACE V5

```
http://www.idealtest.org/dbDetailForUser.do?id=9
```

ARCFACE

```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}

@inproceedings{guo2018stacked,
  title={Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment},
  author={Guo, Jia and Deng, Jiankang and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={BMVC},
  year={2018}
}

@article{deng2018menpo,
  title={The Menpo benchmark for multi-pose 2D and 3D facial landmark localisation and tracking},
  author={Deng, Jiankang and Roussos, Anastasios and Chrysos, Grigorios and Ververas, Evangelos and Kotsia, Irene and Shen, Jie and Zafeiriou, Stefanos},
  journal={IJCV},
  year={2018}
}

@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
```

