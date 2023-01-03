# PACO model zoo

## Introduction

This file lists a collection of models reported in our paper. Each row in tables below includes links to a model and corresponding config as well as object, part, and attribute box AP results and query AR@1 on the specified evaluation dataset.

## Models trained on PACO-LVIS dataset only

Configs listed in the following table by default specify both PACO-LVIS and PACO-EGO4D training datasets. To train only on PACO-LVIS dataset the configs were overridden through the launch command by specifying the `dataloader.train.dataset.names="paco_lvis_v1_train"` option. For example, to train R50 FPN model on PACO-LVIS dataset only, we ran:
```
./tools/lazyconfig_train_net.py --config-file ./configs/mask_rcnn_configs/r50_attr_fpn_100_ep.py --num-gpus 8 dataloader.train.dataset.names=paco_lvis_v1_train
```
from the paco repo root.

| Model | Config | Eval set | AP<sup>obj</sup> | AP<sup>opart</sup> | AP<suB>att</suB><sup>obj</sup> | AP<suB>att</suB><sup>opart</sup> | AR@1 |
|------|------|------|------|------|------|------|------|
| [r50_fpn_lvis](https://dl.fbaipublicfiles.com/paco/models/r50_fpn_lvis.pth) | [r50_attr_fpn_100_ep](../configs/mask_rcnn_configs/r50_attr_fpn_100_ep.py) | PACO-LVIS | 34.7 | 15.8 | 13.0 | 9.9 | 22.4 |
| [r101_fpn_lvis](https://dl.fbaipublicfiles.com/paco/models/r101_fpn_lvis.pth) | [r101_attr_fpn_100_ep](../configs/mask_rcnn_configs/r101_attr_fpn_100_ep.py) | PACO-LVIS | 35.6 | 16.4 | 13.7 | 9.8 | 20.6 |
| [vit_b_fpn_lvis](https://dl.fbaipublicfiles.com/paco/models/vit_b_fpn_lvis.pth) | [vit_b_attr_fpn_100_ep](../configs/mask_rcnn_configs/vit_b_attr_fpn_100_ep.py) | PACO-LVIS | 36.9 | 16.7 | 14.8 | 10.8 | 24.1 |
| [vit_l_fpn_lvis](https://dl.fbaipublicfiles.com/paco/models/vit_l_fpn_lvis.pth) | [vit_l_attr_fpn_100_ep](../configs/mask_rcnn_configs/vit_l_attr_fpn_100_ep.py) | PACO-LVIS | 47.5 | 22.1 | 18.6 | 13.7 | 31.2 |
| [r50_fpn_cascade_lvis](https://dl.fbaipublicfiles.com/paco/models/r50_fpn_cascade_lvis.pth) | [r50_attr_fpn_cascade_100_ep](../configs/mask_rcnn_configs/r50_attr_fpn_cascade_100_ep.py) | PACO-LVIS | 38.6 | 17.2 | 15.6 | 11.0 | 24.2 |
| [r101_fpn_cascade_lvis](https://dl.fbaipublicfiles.com/paco/models/r101_fpn_cascade_lvis.pth) | [r101_attr_fpn_cascade_100_ep](../configs/mask_rcnn_configs/r101_attr_fpn_cascade_100_ep.py) | PACO-LVIS | 40.3 | 18.1 | 16.1 | 11.3 | 25.7 |
| [vit_b_fpn_cascade_lvis](https://dl.fbaipublicfiles.com/paco/models/vit_b_fpn_cascade_lvis.pth) | [vit_b_attr_fpn_cascade_100_ep](../configs/mask_rcnn_configs/vit_b_attr_fpn_cascade_100_ep.py) | PACO-LVIS | 38.2 | 17.5 | 15.7 | 10.8 | 24.4 |
| [vit_l_fpn_cascade_lvis](https://dl.fbaipublicfiles.com/paco/models/vit_l_fpn_cascade_lvis.pth) | [vit_l_attr_fpn_cascade_100_ep](../configs/mask_rcnn_configs/vit_l_attr_fpn_cascade_100_ep.py) | PACO-LVIS | 49.9 | 22.8 | 19.8 | 14.0 | 30.1 |

## Models trained on joint PACO (PACO-LVIS + PACO-EGO4D) dataset

Configs listed in the following table by default specify PACO-LVIS test datasets. To test on PACO-EGO4D dataset the configs were overridden through the launch command by specifying the `dataloader.test.dataset.names="paco_lvis_v1_test"` option. For example, to evaluate ViT-B FPN model on PACO-EGO4D query dataset, we ran:
```
./tools/lazyconfig_train_net.py --config-file ./configs/query_eval_configs/vit_b_fpn_query_eval.py --eval-only --num-gpus 8 train.init_checkpoint=./models/vit_b_fpn_joint.pth dataloader.test.dataset.names=paco_ego4d_v1_test
```
from the paco repo root (assuming the models were downloaded into `./models` folder in the paco repo root).

| Model | Config | Eval set | AP<sup>obj</sup> | AP<sup>opart</sup> | AP<suB>att</suB><sup>obj</sup> | AP<suB>att</suB><sup>opart</sup> | AR@1 |
|------|------|------|------|------|------|------|------|
| [r50_fpn_joint](https://dl.fbaipublicfiles.com/paco/models/r50_fpn_joint.pth) | [r50_attr_fpn_100_ep](../configs/mask_rcnn_configs/r50_attr_fpn_100_ep.py) | PACO-LVIS | 34.6 | 15.8 | 13.8 | 9.8 | 22.1 |
|   |   | PACO-EGO4D | 19.5 | 8.3 | 6.8 | 5.3 | 14.2 |
| [r101_fpn_joint](https://dl.fbaipublicfiles.com/paco/models/r101_fpn_joint.pth) | [r101_attr_fpn_100_ep](../configs/mask_rcnn_configs/r101_attr_fpn_100_ep.py) | PACO-LVIS | 36.0 | 17.0 | 13.9 | 10.3 | 22.8 |
|   |   | PACO-EGO4D | 20.3 | 8.7 | 7.2 | 6.1 | 14.6 |
| [vit_b_fpn_joint](https://dl.fbaipublicfiles.com/paco/models/vit_b_fpn_joint.pth) | [vit_b_attr_fpn_100_ep](../configs/mask_rcnn_configs/vit_b_attr_fpn_100_ep.py) | PACO-LVIS | 36.0 | 17.0 | 13.9 | 10.3 | 28.0 |
|   |   | PACO-EGO4D | 20.3 | 9.9 | 8.6 | 7.3 | 10.7 |
| [vit_l_fpn_joint](https://dl.fbaipublicfiles.com/paco/models/vit_l_fpn_joint.pth) | [vit_l_attr_fpn_100_ep](../configs/mask_rcnn_configs/vit_l_attr_fpn_100_ep.py) | PACO-LVIS | 49.8 | 23.1 | 18.5 | 14.1 | 36.6 |
|   |   | PACO-EGO4D | 30.3 | 14.3 | 12.0 | 10.5 | 22.1 |
| [r50_fpn_cascade_joint](https://dl.fbaipublicfiles.com/paco/models/r50_fpn_cascade_joint.pth) | [r50_attr_fpn_cascade_100_ep](../configs/mask_rcnn_configs/r50_attr_fpn_cascade_100_ep.py) | PACO-LVIS | 38.3 | 17.0 | 15.9 | 11.3 | 24.1 |
|   |   | PACO-EGO4D | 21.8 | 8.9 | 8.4 | 6.7 | 16.0 |
| [r101_fpn_cascade_joint](https://dl.fbaipublicfiles.com/paco/models/r101_fpn_cascade_joint.pth) | [r101_attr_fpn_cascade_100_ep](../configs/mask_rcnn_configs/r101_attr_fpn_cascade_100_ep.py) | PACO-LVIS | 39.8 | 17.9 | 16.4 | 11.6 | 25.1 |
|   |   | PACO-EGO4D | 23.0 | 9.8 | 8.6 | 7.1 | 14.4 |
| [vit_b_fpn_cascade_joint](https://dl.fbaipublicfiles.com/paco/models/vit_b_fpn_cascade_joint.pth) | [vit_b_attr_fpn_cascade_100_ep](../configs/mask_rcnn_configs/vit_b_attr_fpn_cascade_100_ep.py) | PACO-LVIS | 41.7 | 18.9 | 17.0 | 12.2 | 25.9 |
|   |   | PACO-EGO4D | 20.9 | 10.6 | 8.4 | 7.1 | 9.8 |
| [vit_l_fpn_cascade_joint](https://dl.fbaipublicfiles.com/paco/models/vit_l_fpn_cascade_joint.pth) | [vit_l_attr_fpn_cascade_100_ep](../configs/mask_rcnn_configs/vit_l_attr_fpn_cascade_100_ep.py) | PACO-LVIS | 51.7 | 24.3 | 20.1 | 14.6 | 32.0 |
|   |   | PACO-EGO4D | 31.2 | 15.5 | 12.6 | 10.5 | 20.2 |
