# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.utils.file_io import PathManager
from tqdm import tqdm

from .ac_dumper import ACDumper
from .utils import compute_similarity_matrix


class PACOQueryEvalAPI:
    def __init__(
        self,
        query_dicts: List[Dict[str, Any]],
        max_top_k: int = 400,
    ) -> None:
        """
        Provides API for query evaluation per-image processing and final query average
        recall (AR) calculation. AR is calculated by averaging recall for a single
        query over all queries and all IoU thresholds.

        Args:
            query_dicts:    List of query dictionaries (dataset["queries"] field from a
                            query dataset)
            max_top_k:      Maximum k for which recall@k can be requested
        """
        # Extract query info.
        self._extract_query_info(query_dicts)
        self.max_top_k = max_top_k
        self.iou_thrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.reset()

    def reset(self) -> None:
        """
        Resets the state before the per-image processing begins accumulating.
        """
        num_iou_thrs = len(self.iou_thrs)
        self.query_pos_scores = [
            [[] for _ in range(self.num_queries)] for _ in range(num_iou_thrs)
        ]
        self.query_neg_scores = [
            [[] for _ in range(self.num_queries)] for _ in range(num_iou_thrs)
        ]
        self.query_recall_at_k = np.zeros(
            (num_iou_thrs, self.num_queries, self.max_top_k + 1)
        )
        self.results = {}

    def process_image(
        self,
        det_bboxes: np.ndarray,
        det_scores: np.ndarray,
        gt_bboxes: np.ndarray,
        gt_bbox_pos_query_ids: np.ndarray,
        gt_bbox_neg_query_ids: np.ndarray,
        neg_im_gt_query_ids: np.ndarray,
    ) -> None:
        """
        Processes results for a single image. Matches detected bounding boxes with
        GT and accumulates positive and negative scores for each IoU threshold for
        each query.

        Args:
            det_bboxes:             (K, 4) array of detected bounding boxes
            det_scores:             (K, M) array of query scores for each detected
                                    bounding box
            gt_bboxes:              (N, 4) array of GT bounding boxes
            gt_bbox_pos_query_ids:  (N, BPmax) array of positive query IDs for each GT
                                    bounding box
            gt_bbox_neg_query_ids:  (N, BNmax) array of negative query IDs for each GT
                                    bounding box
            neg_im_gt_query_ids:    (IN,) array of query IDs for which this image is
                                    negative
        """
        # Find IoUs between all GT and detected boxes ((N, K) array).
        ious = self.get_bbox_ious(gt_bboxes, det_bboxes)
        for iou_idx, iou_th in enumerate(self.iou_thrs):
            iou_matches = ious >= iou_th
            # Store positive and negative scores for the queries for which this image
            # is positive.
            for box_idx, (pos_query_ids, neg_query_ids) in enumerate(
                zip(gt_bbox_pos_query_ids, gt_bbox_neg_query_ids)
            ):
                # Get detected scores for detected boxes which match with current
                # ground truth box. If there are no matches, skip further processing.
                matched_det_scores = det_scores[iou_matches[box_idx], :]
                if matched_det_scores.size == 0:
                    continue
                # Store score from matched detected box for each positive query ID. If
                # more than one box is matched with GT for the same query ID use the
                # detected box with the highest score. Note also that pos_query_ids
                # may be padded with -1, hence >= 0 check.
                for query_id in pos_query_ids[pos_query_ids >= 0]:
                    scores = matched_det_scores[:, query_id]
                    scores = scores[scores > 0.0]
                    if scores.size > 0:
                        score = scores.max().item()
                        self.query_pos_scores[iou_idx][query_id].append(score)
                # Store score from matched detected box for each negative query ID.
                # Per-box negative query IDs are needed only for positive images for
                # that query, all other negatives are sourced from negative images.
                # Only known negatives in the positive image are accounted for, all
                # other detections are ignored to handle images that are not
                # exhaustively annotated for boxes or queries. Note also that
                # pos_query_ids may be padded with -1, hence >= 0 check.
                for query_id in neg_query_ids[neg_query_ids >= 0]:
                    scores = matched_det_scores[:, query_id]
                    scores = scores[scores > 0.0]
                    if scores.size > 0:
                        score = scores.max().item()
                        self.query_neg_scores[iou_idx][query_id].append(score)
            # Store scores from all detected boxes for query IDs for which this image
            # is negative. The score matrix is very sparse, store only non-zero scores
            # to save memory (needed in case of large number (e.g., 10k) of boxes).
            # The same is done for positive scores so in case all scores are zero the
            # recall will still be 0.
            for query_id in neg_im_gt_query_ids:
                scores = det_scores[:, query_id]
                scores = scores[scores > 0.0].tolist()
                self.query_neg_scores[iou_idx][query_id] += scores

    def evaluate(
        self,
        levels: Optional[List[str]] = None,
        iou_ths: Optional[List[Any]] = None,
        ks: Optional[List[int]] = None,
    ) -> None:
        """
        Calculates recalls for individual queries for all k values after per-image
        processing is done.

        Args:
            levels:     List of query levels for which to calculate average recall,
                        e.g., "all", "l1obj", "l2", etc.
            iou_ths:    List of IoU thresholds for which to calculate average recall,
                        None for all or one of the values in self.iou_thrs
            ks:         List of top K values for which to calculate average recall,
                        an int in [1, self.max_top_k] range
        """
        # Set defaults.
        if levels is None:
            levels = ["l1obj", "l1part"] + [f"l{l}" for l in range(1, 10)] + ["all"]
            levels = [level for level in levels if level in self.query_subsets]
        if iou_ths is None:
            iou_ths = [None, 0.5, 0.75]
        if ks is None:
            ks = [1, 5, 10]
        # Check that all levels are available (IoU thresholds and top K values
        # are checked in _get_average_recall below).
        extra_levels = set(levels) - set(self.query_subsets.keys())
        if len(extra_levels) > 0:
            raise ValueError(
                f"Query levels {extra_levels} not available in the dataset."
            )
        # Caluclate all recalls.
        self._calculate_recalls()
        # Generate results for all levels, IoU thresholds, and top K values.
        for level in levels:
            query_subset = self.query_subsets[level]
            for iou_th in iou_ths:
                for k in ks:
                    case = [f"AR@{k}"]
                    case += [level.upper()]
                    case += [f"IoU{int(iou_th * 100)}" if iou_th is not None else "ALL"]
                    case = "_".join(case)
                    self.results[case] = self._get_average_recall(
                        k=k, query_subset=query_subset, iou_th=iou_th
                    )

    def aggregate_and_set_query_scores(
        self,
        query_pos_scores_chunks: List[List[List[List[float]]]],
        query_neg_scores_chunks: List[List[List[List[float]]]],
    ) -> None:
        """
        Needed for distributed evaluation. Aggregates query positive and negative
        scores accumulated across multiple workers.

        Args:
            query_pos_scores_chunks:    List of positive scores for all IoUs and query
                                        IDs, one for each worker.
            query_neg_scores_chunks:    List of negative scores for all IoUs and query
                                        IDs, one for each worker.
        """
        self.reset()
        for query_pos_scores, query_neg_scores in zip(
            query_pos_scores_chunks, query_neg_scores_chunks
        ):
            for iou_idx in range(len(self.iou_thrs)):
                for query_id in range(self.num_queries):
                    scores = query_pos_scores[iou_idx][query_id]
                    self.query_pos_scores[iou_idx][query_id] += scores
                    scores = query_neg_scores[iou_idx][query_id]
                    self.query_neg_scores[iou_idx][query_id] += scores

    def print_results(self) -> None:
        """
        Prints results from evaluation.
        """
        spaces_top_k = len(str(self.max_top_k))
        spaces_level = max([len(key) for key in self.query_subsets.keys()])
        template = f" {{:<20}} @[topK={{:>{spaces_top_k}d}} | queryLevel={{:>{spaces_level}s}} | IoU={{:<4}}] = {{:0.3f}}"

        for key, value in self.results.items():
            title = "Average Recall (AR)"
            k, level, iou = self._get_params_from_case_string(key)
            print(template.format(title, k, level, iou, value))

    def tabulate_results(self) -> None:
        """
        Tabulate results from evaluation.
        """
        ks = set()
        levels = set()
        ious = set()
        values = defaultdict(lambda: defaultdict(dict))
        for key, value in self.results.items():
            k, level, iou = self._get_params_from_case_string(key)
            ks.add(k)
            levels.add(level)
            ious.add(iou)
            values[k][level][iou] = value
        ks = sorted(ks)
        levels = sorted(levels)
        ious = sorted(ious)
        values = {
            k1: {k2: dict(v2) for k2, v2 in v1.items()} for k1, v1 in values.items()
        }
        spaces = 4 + max([len(key) for key in self.query_subsets.keys()])

        # AR at IoU = all for various top Ks and query levels.
        template = "| {:<10s} |" + "".join(len(levels) * [f" {{:>{spaces}s}} |"])
        iou = "all"
        print(template.format(f"IoU = {iou}", *[f"l = {level}" for level in levels]))
        for k in ks:
            print(
                template.format(
                    f"k = {k}", *[f"{values[k][level][iou]:.3f}" for level in levels]
                )
            )

        # AR at top K = 5 for various IoUs and query levels.
        template = "| {:<10s} |" + "".join(len(levels) * [f" {{:>{spaces}s}} |"])
        k = 5
        print(template.format(f"k = {k}", *[f"l = {level}" for level in levels]))
        for iou in ious:
            print(
                template.format(
                    f"IoU = {iou}",
                    *[f"{values[k][level][iou]:.3f}" for level in levels],
                )
            )

        # AR at level = "all" for various top Ks and IoUs.
        template = "| {:<10s} |" + "".join(len(ks) * [f" {{:>{spaces}s}} |"])
        level = "all"
        print(template.format(f"l = {level}", *[f"k = {k}" for k in ks]))
        for iou in ious:
            print(
                template.format(
                    f"IoU = {iou}", *[f"{values[k][level][iou]:.3f}" for k in ks]
                )
            )

    @staticmethod
    def get_bbox_ious(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
        """
        Calculates IoUs between two sets of boxes.

        Args:
            boxes:       (N, 4) float64 array
            query_boxes: (K, 4) float64 array
        Boxes are specified in (left, top, right, bottom) XYXY format.

        Returns:
            ious: (N, K) float64 array of IoUs between boxes and query_boxes
        """
        # Calculate intersection widths for all (box, query_box) pairs.
        iw = np.minimum(boxes[:, None, 2], query_boxes[None, :, 2])
        iw -= np.maximum(boxes[:, None, 0], query_boxes[None, :, 0])
        iw = np.maximum(iw, 0)
        # Calculate intersection heights for all (box, query_box) pairs.
        ih = np.minimum(boxes[:, None, 3], query_boxes[None, :, 3])
        ih -= np.maximum(boxes[:, None, 1], query_boxes[None, :, 1])
        ih = np.maximum(ih, 0)
        # Calculate intersection areas for all (box, query_box) pairs.
        ious = iw * ih
        # Calculate box areas and their union.
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        query_box_areas = query_boxes[:, 2] - query_boxes[:, 0]
        query_box_areas *= query_boxes[:, 3] - query_boxes[:, 1]
        ua = np.maximum(
            box_areas[:, None] + query_box_areas[None, :] - ious, np.finfo(float).eps
        )
        # Finally, calculate intersection over union.
        ious /= ua
        return ious

    @staticmethod
    def _get_params_from_case_string(case):
        """
        Helper function for result printing.
        """
        k, level, iou = case.split("_")
        k = int(k.replace("AR@", ""))
        level = level.lower()
        if iou == "ALL":
            iou = iou.lower()
        else:
            iou = "{:0.2f}".format(float(iou.replace("IoU", "")) / 100)
        return k, level, iou

    def _calculate_recalls(self):
        """
        Use accumulated positive and negative scores for each query and IoU threshold
        and calculate recall@k for all values of k up to the maximum.
        """
        for iou_idx, (query_pos_scores, query_neg_scores) in enumerate(
            zip(self.query_pos_scores, self.query_neg_scores)
        ):
            for query_id, (pos_scores, neg_scores) in enumerate(
                zip(query_pos_scores, query_neg_scores)
            ):
                num_pos = len(pos_scores)
                if num_pos > 0:
                    scores = np.array(pos_scores + neg_scores)
                    pos_ranks = np.where(scores.argsort(kind="stable")[::-1] < num_pos)
                    pos_ranks = np.minimum(pos_ranks[0], self.max_top_k)
                    self.query_recall_at_k[iou_idx][query_id][pos_ranks] = 1 / num_pos
        self.query_recall_at_k = self.query_recall_at_k.cumsum(axis=-1)

    def _get_average_recall(self, k=1, query_subset=None, iou_th=None):
        """
        Averages recall at specified K across desired subset of queries and
        IoU thresholds.
        """
        if k < 1 or k > self.max_top_k:
            raise ValueError(f"K={k} out of range for recall@K calculation")
        recalls = self.query_recall_at_k[..., k - 1]
        if query_subset is not None:
            recalls = recalls[:, query_subset]
        if iou_th is not None:
            if not np.any(np.isclose(iou_th, self.iou_thrs)):
                raise ValueError(f"IoU threshold {iou_th} not supported.")
            iou_idx = np.where(np.isclose(iou_th, self.iou_thrs))[0]
            recalls = recalls[iou_idx]
        return np.mean(recalls)

    def _extract_query_info(self, query_dicts):
        """
        Extracts query info from query dicts (dataset["queries"] field from a query
        dataset). Populates query_id_to_string, query_subsets and num_queries
        instance variables.
        """
        self.query_id_to_string = {}
        self.query_subsets = defaultdict(list)
        for query_id, d in enumerate(sorted(query_dicts, key=lambda x: x["id"])):
            self.query_id_to_string[query_id] = d["query_string"]
            # Consider query only if it has at least one positive.
            if len(d["pos_ann_ids"]) > 0:
                self.query_subsets["all"].append(query_id)
                self.query_subsets[f"l{d['level']}"].append(query_id)
                # Find whether this is an object only query.
                cat = d["structured_query"][0]
                is_obj = all([thing == cat for _, thing in d["structured_query"][1:]])
                # Populate object/part subsets.
                if is_obj:
                    self.query_subsets["obj"].append(query_id)
                    self.query_subsets[f"l{d['level']}obj"].append(query_id)
                else:
                    self.query_subsets["part"].append(query_id)
                    self.query_subsets[f"l{d['level']}part"].append(query_id)
        self.num_queries = len(self.query_id_to_string)
        self.query_subsets = {k: np.array(v) for k, v in self.query_subsets.items()}


def extract_query_gt_from_dataset(
    dataset: Dict[str, Any]
) -> Tuple[
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
]:
    """
    Extracts per-image query ground truth information needed for evaluation from the
    provided dataset.

    Args:
        dataset:    Dataset dictionary containing "annotations" and "queries" fields

    Returns:
        im_id_to_gt_bboxes:             A map between image ID and an (N, 4) numpy
                                        array containing GT bounding boxes that have
                                        positive or negative query IDs associated with
                                        them.
        im_id_to_gt_bbox_pos_query_ids: A map between image ID and (N, QPmax) numpy
                                        array containing positive query IDs for each
                                        bounding box. QPmax is the maximum number of
                                        positive queries in the image across all boxes.
        im_id_to_gt_bbox_neg_query_ids: A map between image ID and (N, QNmax) numpy
                                        array containing negative query IDs for each
                                        bounding box. QNmax is the maximum number of
                                        negative queries in the image across all boxes.
        im_id_to_neg_im_gt_query_ids:   A map between image ID and a (Q,) numpy array
                                        containing all query IDs for which the image
                                        is negative.
    """
    # Extract map between image ID and annotation IDs for boxes in that image.
    im_id_to_ann_ids = {img["id"]: [] for img in dataset["images"]}
    for ann in dataset["annotations"]:
        im_id_to_ann_ids[ann["image_id"]].append(ann["id"])

    # Extract maps between annotation ID and positive and negative queries for
    # that box.
    ann_id_to_pos_query_ids = defaultdict(list)
    ann_id_to_neg_query_ids = defaultdict(list)
    im_id_to_neg_query_ids = {im_id: [] for im_id in im_id_to_ann_ids.keys()}
    for query in dataset["queries"]:
        for ann_id in query["pos_ann_ids"]:
            ann_id_to_pos_query_ids[ann_id].append(query["id"])
        for ann_id in query["neg_ann_ids"]:
            ann_id_to_neg_query_ids[ann_id].append(query["id"])
        for im_id in query["neg_im_ids"]:
            im_id_to_neg_query_ids[im_id].append(query["id"])
    ann_id_to_pos_query_ids = dict(ann_id_to_pos_query_ids)
    ann_id_to_neg_query_ids = dict(ann_id_to_neg_query_ids)

    # Populate maps used for GT in eval.
    ann_id_to_ann = {ann["id"]: ann for ann in dataset["annotations"]}
    query_ann_ids = set(ann_id_to_pos_query_ids.keys())
    query_ann_ids = query_ann_ids.union(set(ann_id_to_neg_query_ids.keys()))
    im_id_to_gt_bboxes = {}
    im_id_to_gt_bbox_pos_query_ids = {}
    im_id_to_gt_bbox_neg_query_ids = {}
    im_id_to_neg_im_gt_query_ids = {}
    for im_id, ann_ids in im_id_to_ann_ids.items():
        # Keep only boxes that have positive or negative query IDs associated
        # with them (reduces IoU calculation workload during eval).
        ann_ids = [ann_id for ann_id in ann_ids if ann_id in query_ann_ids]
        if len(ann_ids) > 0:
            bboxes = np.array([ann_id_to_ann[ann_id]["bbox"] for ann_id in ann_ids])
            # Convert from XYWH to XYXY.
            bboxes[:, 2:] += bboxes[:, :2]
            im_id_to_gt_bboxes[im_id] = np.array(bboxes)
            # Find positive query IDs for each remaining box. Pad the array with -1
            # to make it square.
            query_ids = [ann_id_to_pos_query_ids.get(ann_id, []) for ann_id in ann_ids]
            max_len = max([len(ids) for ids in query_ids])
            query_ids = [ids + (max_len - len(ids)) * [-1] for ids in query_ids]
            im_id_to_gt_bbox_pos_query_ids[im_id] = np.array(query_ids)
            # Find negative query IDs for each remaining box. Pad the array with -1
            # to make it square.
            query_ids = [ann_id_to_neg_query_ids.get(ann_id, []) for ann_id in ann_ids]
            max_len = max([len(ids) for ids in query_ids])
            query_ids = [ids + (max_len - len(ids)) * [-1] for ids in query_ids]
            im_id_to_gt_bbox_neg_query_ids[im_id] = np.array(query_ids)
        else:
            im_id_to_gt_bboxes[im_id] = np.empty((0, 4))
            im_id_to_gt_bbox_pos_query_ids[im_id] = np.empty((0, 0), dtype=int)
            im_id_to_gt_bbox_neg_query_ids[im_id] = np.empty((0, 0), dtype=int)
        # Populate negative query IDs for the current image.
        im_id_to_neg_im_gt_query_ids[im_id] = np.array(im_id_to_neg_query_ids[im_id])
    return (
        im_id_to_gt_bboxes,
        im_id_to_gt_bbox_pos_query_ids,
        im_id_to_gt_bbox_neg_query_ids,
        im_id_to_neg_im_gt_query_ids,
    )


class PACOQueryPredictionEvaluator:
    def __init__(
        self,
        dataset: Dict[str, Any],
        predictions: List[Dict[str, Any]],
    ):
        """
        Evaluates query detections given in the predictions dump. Predictions dump is
        a list of dictionaries containing predicted boxes and query scores by a
        query detection model for all images in the dataset.
        Args:
            dataset:        Dictionary containing a loaded dataset JSON
            predictions:    List of prediction dictionaries for images in the dataset.
                            Each dictionary contains the following fields:
                            image_id:       int
                            bboxes:         (K, 4) tensor or numpy array of detected
                                            bounding boxes
                            AND EITHER
                            scores:         (K, M) tensor or numpy array of query
                                            scores for each detected bounding box and
                                            each of the M queries
                            OR
                            box_scores:     (K, ) tensor or numpy array of query scores
                                            for each detected bounding box
                            pred_classes:   (K, ) tensor or numpy array of query IDs
                                            for each detected bounding box
        """
        # Check consistency.
        assert "queries" in dataset, "Invalid dataset, missing queries field."
        for prediction in predictions:
            if set(prediction.keys()) != {"image_id", "bboxes", "scores"} and set(
                prediction.keys()
            ) != {"image_id", "bboxes", "box_scores", "pred_classes"}:
                raise ValueError("Invalid predictions dump format.")
        # Instantiate query eval API.
        self.eval_api = PACOQueryEvalAPI(dataset["queries"])
        # Extract maps from dataset and predictions.
        (
            self.im_id_to_gt_bboxes,
            self.im_id_to_gt_bbox_pos_query_ids,
            self.im_id_to_gt_bbox_neg_query_ids,
            self.im_id_to_neg_im_gt_query_ids,
        ) = extract_query_gt_from_dataset(dataset)
        self._build_prediction_maps(predictions)

    def evaluation_loop(
        self, deduplicate_boxes: bool = True, print_results: bool = False
    ):
        """
        Process predictions for all images in the dataset and evaluates
        query prediction performance:

        Args:
            deduplicate_boxes:  Whether to deduplicate boxes or not. Used only when
                                predictions are provided as lists of boxes and
                                predicted class scores, it helps speed up processing
                                in the standard detection frameworks where query
                                predictions share the same boxes. Set to false when
                                boxes are different for each query (e.g., MDETR).
            print_results:      Whether to print results or not
        """
        self.eval_api.reset()
        for im_id, gt_bboxes in tqdm(
            self.im_id_to_gt_bboxes.items(), desc="Processing images"
        ):
            det_bboxes, det_scores = self._prediction_to_det_bboxes_and_scores(
                self.im_id_to_prediction.get(im_id, {}),
                deduplicate_boxes,
                self.eval_api.num_queries,
            )
            gt_bbox_pos_query_ids = self.im_id_to_gt_bbox_pos_query_ids[im_id]
            gt_bbox_neg_query_ids = self.im_id_to_gt_bbox_neg_query_ids[im_id]
            neg_im_gt_query_ids = self.im_id_to_neg_im_gt_query_ids[im_id]
            self.eval_api.process_image(
                det_bboxes,
                det_scores,
                gt_bboxes,
                gt_bbox_pos_query_ids,
                gt_bbox_neg_query_ids,
                neg_im_gt_query_ids,
            )
        self.eval_api.evaluate()
        if print_results:
            self.eval_api.print_results()
            self.eval_api.tabulate_results()

    def get_results(self):
        return self.eval_api.results

    def _build_prediction_maps(self, predictions):
        self.im_id_to_prediction = {}
        for prediction in predictions:
            self.im_id_to_prediction[prediction["image_id"]] = prediction

    def _prediction_to_det_bboxes_and_scores(
        self, prediction, deduplicate_boxes, num_queries, dedup_iou=0.99
    ):
        if "scores" in prediction:
            # Check consistency.
            assert (
                prediction["bboxes"].shape[0] == prediction["scores"].shape[0]
            ), "Number of boxes and scores mismatches"
            assert (
                prediction["scores"].shape[1] == num_queries
            ), "Number of scores different from number of queries."
            # Just convert to numpy if needed.
            det_bboxes = self._to_numpy(prediction["bboxes"])
            det_scores = self._to_numpy(prediction["scores"])
        elif "box_scores" in prediction and "pred_classes" in prediction:
            # Check consistency.
            assert (
                prediction["bboxes"].shape[0] == prediction["box_scores"].shape[0]
            ), "Number of boxes and scores mismatches"
            assert (
                prediction["bboxes"].shape[0] == prediction["pred_classes"].shape[0]
            ), "Number of boxes and labels mismatches"
            # Deduplicate boxes if desired.
            bboxes = self._to_numpy(prediction["bboxes"])
            if deduplicate_boxes:
                num_unique_bboxes = 0
                bbox_ids = -1 * np.ones(len(bboxes), dtype=int)
                for idx, bbox in enumerate(bboxes):
                    if bbox_ids[idx] == -1:
                        ious = PACOQueryEvalAPI.get_bbox_ious(bbox[None, :], bboxes)
                        bbox_ids[ious[0] > dedup_iou] = num_unique_bboxes
                        num_unique_bboxes += 1
                unique_bboxes = np.empty((num_unique_bboxes, 4))
                unique_bboxes[bbox_ids, :] = bboxes[np.arange(len(bboxes)), :]
            else:
                bbox_ids = np.arange(len(bboxes))
                unique_bboxes = bboxes
            # Build the NxM score array, N being the number of unique boxes.
            box_scores = self._to_numpy(prediction["box_scores"])
            pred_classes = self._to_numpy(prediction["pred_classes"])
            scores = np.zeros_like(box_scores, shape=(len(unique_bboxes), num_queries))
            scores[bbox_ids, pred_classes] = box_scores
            det_bboxes = unique_bboxes
            det_scores = scores
        else:
            det_bboxes = np.empty((0, 4))
            det_scores = np.empty((0, num_queries))
        return det_bboxes, det_scores

    @staticmethod
    def _to_numpy(arr):
        return arr if isinstance(arr, np.ndarray) else arr.numpy()


class PACOQueryEvaluator(ACDumper):
    def __init__(self, dataset_name: str, **kwargs) -> None:
        """
        Evaluates query detections given atomic constructs (objects, parts, and
        attributes) predictions.

        Args:
            dataset_name:   A registered dataset name
        """
        super(PACOQueryEvaluator, self).__init__(dataset_name, **kwargs)

        # Load the dataset and check consistency.
        with PathManager.open(self._metadata.json_file) as f:
            dataset = json.load(f)
        assert "queries" in dataset, "Invalid dataset, missing queries field."

        # Instantiate query eval API.
        self.eval_api = PACOQueryEvalAPI(dataset["queries"])

        # Extract maps from dataset.
        (
            self.im_id_to_gt_bboxes,
            self.im_id_to_gt_bbox_pos_query_ids,
            self.im_id_to_gt_bbox_neg_query_ids,
            self.im_id_to_neg_im_gt_query_ids,
        ) = extract_query_gt_from_dataset(dataset)

        # Construct query features.
        self.query_features = self._construct_query_features(dataset)

    def process(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Processes prediction outputs for one image and accumulates positive/negative
        scores to be used in evaluation.
        Args:
            inputs:     The inputs to an LVIS model (e.g., GeneralizedRCNN). It is a
                        list of dicts. Each dict corresponds to an image and contains
                        keys like "height", "width", "file_name", "image_id".
            outputs:    The outputs of an LVIS model. It is a list of dicts with key
                        "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            if "instances" in output:
                im_id = input["image_id"]
                instances = output["instances"]

                # Extract atomic construct features.
                ac = self.process_instances(im_id, instances)

                # Convert features to query scores.
                if ac["features"].numel() == 0:
                    det_scores = np.empty((0, self.query_features.shape[0]))
                else:
                    det_scores = (
                        compute_similarity_matrix(
                            ac["features"].to(device=self.query_features.device),
                            self.query_features,
                            n_obj=self.n_objs,
                            n_part=self.n_parts,
                            n_attr=self.n_attrs,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

                # Accumulate scores for the current image.
                det_bboxes = ac["bboxes"].detach().cpu().numpy()
                gt_bboxes = self.im_id_to_gt_bboxes[im_id]
                gt_bbox_pos_query_ids = self.im_id_to_gt_bbox_pos_query_ids[im_id]
                gt_bbox_neg_query_ids = self.im_id_to_gt_bbox_neg_query_ids[im_id]
                neg_im_gt_query_ids = self.im_id_to_neg_im_gt_query_ids[im_id]
                self.eval_api.process_image(
                    det_bboxes,
                    det_scores,
                    gt_bboxes,
                    gt_bbox_pos_query_ids,
                    gt_bbox_neg_query_ids,
                    neg_im_gt_query_ids,
                )

    def evaluate(self) -> None:
        """
        Processes accumulated scores and produces AR@k results.
        """
        if self._distributed:
            # Gather accumulated scores.
            comm.synchronize()
            query_pos_scores = comm.gather(self.eval_api.query_pos_scores, dst=0)
            query_neg_scores = comm.gather(self.eval_api.query_neg_scores, dst=0)

            if not comm.is_main_process():
                return {}

            # Aggregate accumulated scores in the main process.
            self.eval_api.aggregate_and_set_query_scores(
                query_pos_scores, query_neg_scores
            )

        # Evaluate.
        self.eval_api.evaluate()
        self.eval_api.print_results()
        self.eval_api.tabulate_results()

        # Return a copy of results so the caller can do whatever with them.
        return {"query": copy.deepcopy(self.eval_api.results)}

    def _construct_query_features(
        self, dataset: Dict[str, Any]
    ) -> Tuple[torch.tensor, int, int, int]:
        """
        Constructs query feature vector from corresponding structured query and
        concatenates vectors for all queries into an (M, D) tensor.
        """
        # Extract maps needed for feature vector construction from dataset.
        cat_name_to_idx = {
            d["name"]: idx
            for idx, d in enumerate(
                sorted(dataset["categories"], key=lambda x: x["id"])
            )
            if ":" not in d["name"]
        }
        part_name_to_idx = {d["name"]: d["id"] for d in dataset["part_categories"]}
        attr_name_to_idx = {d["name"]: d["id"] for d in dataset["attributes"]}

        # Construct query feature vector for each query.
        query_features = []
        for d in dataset["queries"]:
            # Extract object, object attributes, parts, and part attributes from the
            # query.
            query = d["structured_query"]
            cat = query[0]
            cat_attrs = []
            part_to_attrs = defaultdict(list)
            for attr, thing in query[1:]:
                if thing == cat:
                    cat_attrs.append(attr)
                else:
                    part_to_attrs[thing].append(attr)
            # Initialize feature vector.
            feature_vec = torch.zeros((self.feat_dim,), device="cuda")
            # Set feature vector element corresponding to the category.
            cat_idx = cat_name_to_idx[cat]
            feature_vec[cat_idx] = 1
            # Set feature vector element corresponding to all parts that are present.
            part_idxs = [
                self.n_objs + part_name_to_idx[part] for part in part_to_attrs.keys()
            ]
            feature_vec[part_idxs] = 1
            # Set feature vector element corresponding to object attributes present.
            attr_idxs = [
                self.n_objs + self.n_parts + attr_name_to_idx[attr]
                for attr in cat_attrs
            ]
            feature_vec[attr_idxs] = 1
            # Set feature vector element corresponding to part attributes present.
            for part, attrs in part_to_attrs.items():
                part_idx = part_name_to_idx[part]
                offset = self.n_objs + self.n_parts + (part_idx + 1) * self.n_attrs
                attr_idxs = [offset + attr_name_to_idx[attr] for attr in attrs]
                feature_vec[attr_idxs] = 1
            # Append to the output.
            query_features.append(feature_vec[None, :])

        # Concatenate all feature vectors.
        query_features = torch.cat(query_features)

        return query_features
