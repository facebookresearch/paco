# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import (
    FastRCNNConvFCHead,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
)
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler
from paco.evaluation.paco_evaluation import PACOEvaluator
from paco.models.roi_heads import PACOROIHeads

from .coco_lsj_loader import dataloader

dataloader.train.dataset.names = ("paco_joint_v1_train", "paco_ego4d_v1_train")
dataloader.test.dataset.names = "paco_lvis_v1_val"
dataloader.train.mapper.instance_mask_format = "bitmask"
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(
        RepeatFactorTrainingSampler.repeat_factors_from_category_frequency
    )(dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001)
)

dataloader.evaluator = L(PACOEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=300,
)

model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model.roi_heads.update(
    _target_=PACOROIHeads,
    num_classes=531,
    batch_size_per_image=512,
    positive_fraction=0.25,
    proposal_matcher=L(Matcher)(
        thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
    ),
    box_in_features=["p2", "p3", "p4", "p5"],
    box_pooler=L(ROIPooler)(
        output_size=7,
        scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        sampling_ratio=0,
        pooler_type="ROIAlignV2",
    ),
    box_head=L(FastRCNNConvFCHead)(
        input_shape=ShapeSpec(channels=256, height=7, width=7),
        conv_dims=[256, 256, 256, 256],
        fc_dims=[1024],
        conv_norm="LN",
    ),
    box_predictor=L(FastRCNNOutputLayers)(
        input_shape=ShapeSpec(channels=1024),
        test_score_thresh=0.02,
        box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
        num_classes="${..num_classes}",
        test_topk_per_image=300,
        use_sigmoid_ce=True,
        use_fed_loss=True,
        get_fed_loss_cls_weights=lambda: get_fed_loss_cls_weights(
            dataloader.train.dataset.names, 0.5
        ),
    ),
    mask_in_features=["p2", "p3", "p4", "p5"],
    mask_pooler=L(ROIPooler)(
        output_size=14,
        scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        sampling_ratio=0,
        pooler_type="ROIAlignV2",
    ),
    mask_head=L(MaskRCNNConvUpsampleHead)(
        input_shape=ShapeSpec(channels=256, width=14, height=14),
        num_classes="${..num_classes}",
        conv_dims=[256, 256, 256, 256, 256],
        conv_norm="LN",
    ),
)

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth"

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[138889, 150463],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7
)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
optimizer.lr = 2e-4


# Schedule
# 100 ep = 156250 iters * 64 images/iter / 100000 images/ep
train.max_iter = 156250
train.eval_period = 300000
