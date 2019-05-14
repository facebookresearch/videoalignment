# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch


def get_device(model):
    p = next(model.parameters(), None)
    default = "cuda" if torch.cuda.is_available() else "cpu"
    return p.device if p is not None else default
