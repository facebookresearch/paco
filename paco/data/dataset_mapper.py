# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils as utils, transforms as T
from detectron2.data.dataset_mapper import DatasetMapper

from detectron2.structures import BoxMode


logger = logging.getLogger(__name__)

ATTR_TYPE_END_IDXS = [0, 30, 41, 55, 59]
ATTR_TYPE_BG_IDXS = [29, 38, 54, 58]


def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Borrowed from https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/detection_utils.py#L257
    with support for attributes
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(
        annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS
    )
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray]"
                ", COCO-style RLE as a dict.".format(type(segm))
            )

    # support for attributes
    if "attr_labels" in annotation:
        attr_label_tensor = np.zeros(ATTR_TYPE_END_IDXS[-1])
        attr_ignore_tensor = np.zeros(len(ATTR_TYPE_END_IDXS) - 1)
        for _a in annotation["attr_labels"]:
            attr_label_tensor[_a] = 1.0

        for _aid, _a in enumerate(annotation["attr_ignores"]):
            attr_ignore_tensor[_aid] = _a

        for attr_type_id in range(1, len(ATTR_TYPE_END_IDXS)):
            st_idx = ATTR_TYPE_END_IDXS[attr_type_id - 1]
            end_idx = ATTR_TYPE_END_IDXS[attr_type_id]
            bg_idx = ATTR_TYPE_BG_IDXS[attr_type_id - 1]
            attr_label_tensor[st_idx:end_idx] /= max(
                attr_label_tensor[st_idx:end_idx].sum(), 1.0
            )
            attr_label_tensor[bg_idx] = 1.0 - (
                attr_label_tensor[st_idx:end_idx].sum() - attr_label_tensor[bg_idx]
            )

        annotation["attr_label_tensor"] = attr_label_tensor
        annotation["attr_ignore_tensor"] = attr_ignore_tensor

    return annotation


class PACODatasetMapper(DatasetMapper):
    """
    Modified PACODatasetMapper dataset mapper
    """

    @configurable
    def __init__(self, **kwargs):
        """
        Args: Check DatasetMapper
        """
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train=is_train)
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)

        # USER: Implement additional transformations if you have other types of
        # data
        annos = [
            transform_instance_annotations(
                obj,
                transforms,
                image_shape,
                keypoint_hflip_indices=self.keypoint_hflip_indices,
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )
        if len(annos) and "attr_label_tensor" in annos[0]:
            attr_label_tensor = torch.tensor(
                [obj["attr_label_tensor"] for obj in annos]
            )
            attr_ignore_tensor = torch.tensor(
                [obj["attr_ignore_tensor"] for obj in annos], dtype=torch.int64
            )
            instances.gt_attr_label_tensor = attr_label_tensor
            instances.gt_attr_ignore_tensor = attr_ignore_tensor
        # After transforms such as cropping are applied, the bounding box may
        # no longer tightly bound the object. As an example, imagine a
        # triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format).
        # The tight bounding box of the cropped triangle should be
        # [(1,0),(2,1)], which is not equal to the intersection of original
        # bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
