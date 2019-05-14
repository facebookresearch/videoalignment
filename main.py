# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle as pkl
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve
from videoalignment import datasets, eval, models


def test(model, args):
    outs = defaultdict(list)
    if args.dataset_test in [datasets.Madonna, datasets.EVVE]:
        outs["map"].append(
            eval.map(
                model,
                args.dataset_test,
                args,
                "all",
                query_expansion=args.query_expansion,
            )
        )

    if args.dataset_test.is_localization and args.dataset_test != datasets.VCDB:
        outs["loc_errors"].append(
            eval.localization_errors(model, args.dataset_test, args, "all")
        )

    if args.dataset_test == datasets.VCDB:
        probas, labels = eval.segment_pr(model, args.dataset_test, args, "all")
        outs["precision_recall"].append([probas, labels])
        precision, recall, _ = precision_recall_curve(labels, probas)
        print("AUC test", auc(recall, precision))
        max_f1 = np.max(2 * (precision * recall) / (precision + recall + 10e-8))
        print("MaxF1 test", max_f1)

    output_path = os.path.join(
        args.output_dir, "tests_%s.pkl" % args.dataset_test.__name__
    )
    with open(output_path, "wb") as pfile:
        pkl.dump(outs, pfile, pkl.HIGHEST_PROTOCOL)


def main(args):
    if args.model == "TMK_Poullot":
        args.normalization = "freq"

    excluded = {"output_dir", "pca_mean", "pca_DVt"}
    parameter_string = "_".join(
        ["%s-%s" % (k, str(v)) for (k, v) in vars(args).items() if k not in excluded]
    )
    output_dir = os.path.join(args.output_dir, parameter_string)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(args)
    print("Parameter string is", parameter_string)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.dataset_test = getattr(datasets, args.dataset_test)
    args.model = getattr(models, args.model)

    # TMK layers setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = args.model(args).to(device)
    test(model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video alignment model evaluation")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument(
        "--output_dir", required=True, type=str, help="Output directory for results"
    )
    parser.add_argument("--dataset_test", required=True, type=str, default=None)
    parser.add_argument("--fold_index", required=False, default=0, type=int)
    parser.add_argument("--chunk_randomly", default=1, type=int)
    parser.add_argument("--T", nargs="+", type=int, default=[9767, 2731, 1039, 253])
    parser.add_argument("--resnet_level", default=29, type=int)
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--b_s", default=16, type=int, help="Batch size per GPU")
    parser.add_argument("--m", default=16, type=int, help="TMK parameter m")
    parser.add_argument("--tmk_init", default="von_mises", type=str)
    parser.add_argument("--norm", default="feat_freq", type=str)
    parser.add_argument("--use_pca", default=0, type=int)
    parser.add_argument(
        "--query_expansion",
        default="",
        type=str,
        choices=["", "don", "aqe"],
        help="Strategy for query expansion",
    )
    parser.add_argument(
        "--pca_mean", default="", type=str, help="Path to PCA mean t7 file"
    )
    parser.add_argument(
        "--pca_DVt", default="", type=str, help="Path to PCA DVt t7 file"
    )
    parser.add_argument("--multiple_pca", default=0, type=int)
    args = parser.parse_args()

    main(args)
