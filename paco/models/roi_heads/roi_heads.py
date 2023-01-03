# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
from detectron2.config import configurable
from detectron2.layers import nonzero_tuple, ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import build_box_head, StandardROIHeads
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage

from .attribute_head import AttributeOutputLayers


logger = logging.getLogger(__name__)


# this is added for backwards compatibility with the yaml config options.
# the codebase doesn't explicitly support the yaml config option. User can
# create their own config file to be able to update this through yaml
MAPPED_ATTRS_LIST = [
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
]


def select_proposals_with_attributes(
    proposals: List[Instances],
    bg_label: int,
) -> List[Instances]:
    """
    Given a list of N Instances (for N images), return a list of Instances that
    contain only instances with `gt_classes != -1 && gt_classes != bg_label`.
    In addition, the returned proposals have at-least one non-ignored attribute
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.
        bg_label: label index of background class.
    Returns:
        proposals: only contains proposals with at least one
        non-ignored attribute.

    """
    ret = []
    all_num_valid_attrs = []
    ignored_sum = []
    for _i, proposals_per_image in enumerate(proposals):
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            all_num_valid_attrs.append(0)
            ret.append(proposals_per_image)
            continue

        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        proposals_per_image = proposals_per_image[fg_selection_mask]

        if len(proposals_per_image) == 0:
            all_num_valid_attrs.append(0)
            ret.append(proposals_per_image)
            continue
        # attr_label_tensor is num_proposals X overall attr. label size
        # attr_ignore_tensor is num_proposals X num_attr_types
        assert proposals_per_image.has("gt_classes")
        assert proposals_per_image.has("gt_attr_label_tensor")
        assert proposals_per_image.has("gt_attr_ignore_tensor")

        attr_ignore_tensor = proposals_per_image.gt_attr_ignore_tensor
        # At-least one attribute type is not ignored
        selection = attr_ignore_tensor.sum(dim=1) < attr_ignore_tensor.shape[1]
        selection_idxs = nonzero_tuple(selection)[0]
        all_num_valid_attrs.append(selection_idxs.numel())

        if len(selection_idxs) > 0:
            valid_attr_ids = nonzero_tuple(attr_ignore_tensor[selection_idxs] > 0)[0]
            ignored_sum.append(valid_attr_ids.numel())
        else:
            ignored_sum.append(0.0)
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("attribute/num_valid_attrs", np.mean(all_num_valid_attrs))
    storage.put_scalar("attribute/num_ignored", np.mean(ignored_sum))
    return ret


@ROI_HEADS_REGISTRY.register()
class PACOROIHeads(StandardROIHeads):
    """
    Inherits from StandardROIHeads and adds support for attribute head
    """

    @configurable
    def __init__(
        self,
        *,
        attr_in_features: Optional[List[str]] = None,
        attr_pooler: Optional[ROIPooler] = None,
        attr_head: Optional[nn.Module] = None,
        attr_predictor: nn.Module = None,
        **kwargs,
    ):
        """
        Args:
            attr_in_features (list[str]): list of feature names to use for the
                attr head.
            attr_pooler (ROIPooler): pooler to extra region features for the
                attr head
            attr_head (nn.Module): transform features to make attr predictions
            attr_predictor (nn.Module): make attr predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`
        """
        super().__init__(**kwargs)

        self.attributes_on = attr_in_features is not None
        if self.attributes_on:
            self.attr_in_features = attr_in_features
            self.attr_pooler = attr_pooler
            self.attr_head = attr_head
            self.attr_predictor = attr_predictor

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        # Subclasses that have not been updated to use from_config style
        # construction may have overridden _init_*_head methods. In this case,
        # those overridden methods will not be classmethods and we need to
        # avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound
        # methods of cls. Such subclasses will need to handle calling their
        # overridden _init_*_head methods.

        if inspect.ismethod(cls._init_attribute_head):
            ret.update(cls._init_attribute_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_attribute_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts
        # must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        attr_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        attr_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )

        attr_predictor = AttributeOutputLayers(
            cfg, attr_head.output_shape, MAPPED_ATTRS_LIST
        )

        return {
            "attr_in_features": in_features,
            "attr_pooler": attr_pooler,
            "attr_head": attr_head,
            "attr_predictor": attr_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Add call to attributes head
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by
            # the mask, keypoint heads. But when `self.train_on_pred_boxes is
            # True`, proposals will contain boxes predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_attributes(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and
            # keypoints heads are only applied to the top scoring box
            # detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Add call to attributes head
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_attributes(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_attributes(
        self,
        features: Dict[str, torch.Tensor],
        instances: List[Instances],
    ):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to
                tensor. Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to
                train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and
            return it.
        """
        if not self.attributes_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances = select_proposals_with_attributes(
                instances,
                self.num_classes,
            )

        features = [features[f] for f in self.attr_in_features]
        boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
        features = self.attr_pooler(features, boxes)
        features = self.attr_head(features)
        return self.attr_predictor(features, instances)
