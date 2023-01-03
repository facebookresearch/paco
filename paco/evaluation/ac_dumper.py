# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import logging
import os
from collections import defaultdict, OrderedDict

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.data import MetadataCatalog

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.file_io import PathManager


def get_matches(obj_masks, part_masks, thresh):
    """
    for (N x H x W) dim obj masks and (M x H x W) dim part masks, returns
    (N X M) dim bool array where the element at (i, j) stores True if the
    overlap b/w object i and oject-part j crosses the thresh
    """
    intersections = torch.tensordot(
        obj_masks.to(dtype=torch.float32),
        part_masks.to(dtype=torch.float32),
        dims=2 * ([1, 2],),
    )
    part_mask_areas = part_masks.sum(dim=[1, 2]) + 1e-7
    matches_prec = intersections / part_mask_areas[None, ...]
    matches_prec = matches_prec > thresh
    return matches_prec


class ACDumper(DatasetEvaluator):
    """
    Evaluate object proposal and instance detection/segmentation outputs using
    LVIS's metrics and evaluation API.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                "thing_classes": list of object classes
                "part_classes": list of semantic part classes
                "attribute_classes": list of attribute classes
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
            max_dets_per_image (None or int): limit on maximum detections per image in evaluating AP
                This limit, by default of the LVIS dataset, is 300.
        """
        self._logger = logging.getLogger(__name__)

        self._distributed = distributed
        if output_dir:
            self._output_dir = os.path.join(output_dir, dataset_name)
        else:
            self._output_dir = None
        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

        self.n_objs = len(self._metadata.thing_classes)
        self.n_parts = len(self._metadata.part_classes)
        self.n_attrs = len(self._metadata.attribute_classes)
        self.feat_dim = self.n_objs + self.n_parts + (1 + self.n_parts) * self.n_attrs

        self.semantic_part_names_to_ids = {
            x: _i for _i, x in enumerate(self._metadata.part_classes)
        }
        self.obj_name_to_id = {
            x: _i for _i, x in enumerate(self._metadata.thing_classes) if ":" not in x
        }
        self.obj_part_id_to_semantic_id = {
            _i: self.semantic_part_names_to_ids[x.split(":")[-1]]
            for _i, x in enumerate(self._metadata.thing_classes)
            if ":" in x
        }

        # store the object-part categories corressponding to each object cat
        self.obj_to_obj_part_ids = defaultdict(set)
        for _i, x in enumerate(self._metadata.thing_classes):
            if ":" in x:
                obj_cat = x.split(":")[0]
                self.obj_to_obj_part_ids[self.obj_name_to_id[obj_cat]].add(_i)
        # Test set json files do not contain annotations (evaluation must be
        # performed using the LVIS evaluation server).

    def reset(self):
        self._predictions = []

    def process_instances(self, image_id, instances):
        device = instances.pred_masks.device

        # separate predictictions into objects and parts
        is_obj = torch.zeros(len(instances), dtype=bool, device=device)
        for _i, cat_id in enumerate(instances.pred_classes):
            if cat_id.item() < len(self.obj_name_to_id):
                is_obj[_i] = True

        obj_masks = instances.pred_masks[is_obj]
        num_boxes = len(obj_masks)
        obj_boxes = instances.pred_boxes[is_obj].tensor
        obj_scores = instances.scores[is_obj]
        obj_cats = instances.pred_classes[is_obj]
        obj_attrs = instances.attribute_probs[is_obj]

        part_masks = instances.pred_masks[is_obj == 0]
        part_scores = instances.scores[is_obj == 0]
        part_cats = instances.pred_classes[is_obj == 0]
        part_attrs = instances.attribute_probs[is_obj == 0]

        matching_part_ids = set()
        feats = []
        boxes = []
        for cat in torch.unique(obj_cats):
            # for each object cat, select parts where predicted object-part
            # categories contain the object cat
            valid_part_ids = defaultdict(list)
            rel_obj_ids = obj_cats == cat
            rel_obj_masks = obj_masks[rel_obj_ids]

            for rel_joint_cat in self.obj_to_obj_part_ids[cat.item()]:
                # find best matches for this obj cat and object-part cat
                rel_part_anns = part_cats == rel_joint_cat
                if torch.sum(rel_part_anns) > 0:
                    rel_part_masks = part_masks[rel_part_anns]
                    # ids for parts in the original part tensor
                    part_ann_ids = torch.where(rel_part_anns)[0]
                    matches = get_matches(rel_obj_masks, rel_part_masks, thresh=0.5)
                    # we want a bike object to match only with one bike-wheel,
                    # so only retain top scoring matches per object which cross
                    # thresh
                    rel_part_scores = part_scores[rel_part_anns]
                    rel_part_scores_expanded = rel_part_scores[None, ...].repeat(
                        len(rel_obj_masks), 1
                    )
                    rel_part_scores_expanded = rel_part_scores_expanded * matches
                    vals, ids = torch.topk(rel_part_scores_expanded, k=1)
                    for _i, (val, part_idx) in enumerate(zip(vals, ids)):
                        # score must be > 0 for a true match
                        if val > 0:
                            # save id in original part tensor
                            valid_part_ids[_i].append(part_ann_ids[part_idx].item())
                            matching_part_ids.add(part_ann_ids[part_idx])

            # after getting matches, prcocess these boxes and parts to features
            rel_obj_scores = obj_scores[rel_obj_ids]
            rel_obj_attrs = obj_attrs[rel_obj_ids]
            rel_obj_boxes = obj_boxes[rel_obj_ids]

            # create feature for each box
            for obj_id in range(len(rel_obj_scores)):
                part_ids = valid_part_ids[obj_id]
                feat = torch.zeros(self.feat_dim, device=device)

                # populate all object related features first
                boxes.append(rel_obj_boxes[obj_id])  # box
                feat[cat] = rel_obj_scores[obj_id]  # box score
                obj_attr_offset = self.n_objs + self.n_parts
                feat[obj_attr_offset : obj_attr_offset + self.n_attrs] = rel_obj_attrs[
                    obj_id
                ]  # obj attr

                if len(part_ids) > 0:
                    part_scores_for_box = part_scores[part_ids]
                    # map object-part cats to original part cats in semantic cats
                    part_cats_for_box = []
                    for x in part_cats[part_ids]:
                        part_cats_for_box.append(
                            self.obj_part_id_to_semantic_id[x.item()]
                        )
                    part_cats_for_box = torch.tensor(
                        part_cats_for_box, dtype=torch.long, device=device
                    )
                    # part scores
                    feat[self.n_objs + part_cats_for_box] = part_scores_for_box

                    # part attr scores
                    part_attr_scores = part_attrs[part_ids]
                    part_attr_offset = self.n_objs + self.n_parts + self.n_attrs
                    offset_part_attr_inds = (
                        part_attr_offset + part_cats_for_box * self.n_attrs
                    )
                    inds_scatter = np.linspace(
                        offset_part_attr_inds.cpu(),
                        (offset_part_attr_inds + self.n_attrs).cpu(),
                        num=self.n_attrs,
                        endpoint=False,
                        dtype=int,
                    ).T
                    inds_scatter = torch.as_tensor(inds_scatter, device=device)
                    inds_scatter = inds_scatter.reshape(-1)
                    feat[inds_scatter] = part_attr_scores.reshape(-1)

                feats.append(feat)

        if len(feats):
            boxes = torch.stack(boxes)
            feats = torch.stack(feats)
            # store after sorting based on object score
            obj_feats = feats[:, : len(self.obj_name_to_id)]
            sorted_ids = torch.argsort(obj_feats[obj_feats > 0], descending=True)
            assert len(sorted_ids) == len(boxes), "not 1-hot"
            ac = {
                "image_id": image_id,
                "bboxes": boxes[sorted_ids],
                "features": feats[sorted_ids],
            }

            assert len(boxes) == num_boxes, print(num_boxes, len(boxes))
            assert len(boxes) == len(feats)
        else:
            ac = {
                "image_id": image_id,
                "bboxes": torch.empty(0, 4, device=device),
                "features": torch.empty(0, self.feat_dim, device=device),
            }

        return ac

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a LVIS model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a LVIS model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                import time

                st = time.time()
                ac = self.process_instances(input["image_id"], instances)
                en = time.time()
                print(f"time taken: {en -st}")
                if ac is not None:
                    self._predictions.append(ac)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[ACDumper] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        # add empty results for compatibilty
        self._results = OrderedDict()

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
