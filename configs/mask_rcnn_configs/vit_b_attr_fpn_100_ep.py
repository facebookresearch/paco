# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from detectron2.config import LazyCall as L

from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import FastRCNNConvFCHead
from paco.models.roi_heads.attribute_head import AttributeOutputLayers

from .vit_b_fpn_100_ep import dataloader, lr_multiplier, model, optimizer, train

dataloader.train.mapper.instance_mask_format = "bitmask"
dataloader.evaluator.eval_attributes = True

model.roi_heads.attr_in_features = ["p2", "p3", "p4", "p5"]
model.roi_heads.attr_pooler = L(ROIPooler)(
    output_size=7,
    scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
    sampling_ratio=0,
    pooler_type="ROIAlignV2",
)

model.roi_heads.attr_head = L(FastRCNNConvFCHead)(
    input_shape=ShapeSpec(channels=256, height=7, width=7),
    conv_dims=[],
    fc_dims=[1024, 1024],
)
model.roi_heads.attr_predictor = L(AttributeOutputLayers)(
    input_shape=ShapeSpec(channels=1024),
    mapped_ids_list=[
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
        ],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        [55, 56, 57, 58],
    ],
    loss_weight=0.01,
)

train.eval_period = 3000000
optimizer.lr = 2e-4


lr_multiplier.scheduler.milestones = [138889, 150463]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter
