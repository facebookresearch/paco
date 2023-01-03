# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from detectron2.config import LazyCall as L
from paco.evaluation.paco_query_evaluation import PACOQueryEvaluator

from ..mask_rcnn_configs.vit_l_attr_fpn_100_ep import (  # noqa
    dataloader,
    model,
    train,
    lr_multiplier,
    optimizer,
)

dataloader.test.dataset.names = "paco_lvis_v1_test"
dataloader.evaluator = L(PACOQueryEvaluator)(
    dataset_name="${..test.dataset.names}",
)
