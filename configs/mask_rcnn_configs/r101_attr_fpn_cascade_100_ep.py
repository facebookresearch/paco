# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .r50_attr_fpn_cascade_100_ep import (  # noqa
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

model.backbone.bottom_up.stages.depth = 101

train.eval_period = 5000000
