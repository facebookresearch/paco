# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import numpy as np
import os
import torch

from paco.data.datasets.builtin import _PREDEFINED_PACO
from paco.evaluation.paco_query_evaluation import PACOQueryPredictionEvaluator


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description="Query evaluation from predictions dump"
    )
    parser.add_argument(
        "--predictions_file_name",
        help="Path to a predictions dump .pth or .npy file that can be loaded using "
        "torch.load() or numpy.load(). See PACOQueryPredictionEvaluator docstring "
        "for the predictions dump format.",
        required=True,
    )
    parser.add_argument(
        "--dataset_file_name",
        help="Path to a query dataset.",
    )
    parser.add_argument(
        "--deduplicate_boxes",
        help="Whether to deduplicate boxes in predictions. See "
        "PACOQueryPredictionEvaluator.evaluation_loop() docstring.",
    )
    return parser


if __name__ == "__main__":
    # Parse the arguments and set defaults for optional arguments.
    args = get_arg_parser().parse_args()
    if args.dataset_file_name is None:
        if "ego4d" in args.predictions_file_name:
            dataset_file_name = _PREDEFINED_PACO["paco_ego4d_v1_test"][0]
        else:
            dataset_file_name = _PREDEFINED_PACO["paco_lvis_v1_test"][0]
    if args.deduplicate_boxes is None:
        args.deduplicate_boxes = "mdetr" not in args.predictions_file_name
    print(args)

    # Load the dataset.
    print("Loading the dataset...")
    with open(dataset_file_name) as f:
        dataset = json.load(f)
    print(len(dataset["annotations"]))

    # Load the predictions.
    print("Loading predictions...")
    ext = os.path.splitext(args.predictions_file_name)[-1]
    if ext == ".pth":
        predictions = torch.load(args.predictions_file_name)
    elif ext == ".npy":
        predictions = np.load(args.predictions_file_name, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported input file format {ext}, only .pth or .npy are supported.")

    # Instantiate the evaluator.
    qeval = PACOQueryPredictionEvaluator(dataset, predictions)

    # Run the eval loop which will print the results.
    qeval.evaluation_loop(deduplicate_boxes=args.deduplicate_boxes, print_results=True)
