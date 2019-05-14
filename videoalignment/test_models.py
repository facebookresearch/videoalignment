# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import List, NamedTuple

import torch
from videoalignment.models import CTE, TMK, TMK_Poullot


class ModelArgs(NamedTuple):
    m: int = 16
    norm: str = "freq"
    T: List[int] = None
    use_pca: bool = False


def fake_data(d, n_frames_a, n_frames_b, offset):
    """
    Generate fake images descriptors.
    Video A is included in Video B, after an offset.

    |.. before .. | .. video A ..| .. after ..|
    |................. video B ...............|
    -------------------------------------------> time
    """
    torch.manual_seed(10)
    frame_features_a = torch.randn(1, n_frames_a, d).float()
    timestamps_a = torch.arange(0, n_frames_a).float().view(1, -1)

    after = n_frames_b - n_frames_a - offset
    before_features = torch.randn(1, offset, d).float()
    after_features = torch.randn(1, after, d).float()
    frame_features_b = torch.cat(
        [before_features, frame_features_a, after_features], dim=1
    )
    timestamps_b = torch.arange(0, n_frames_b).float().view(1, -1)
    return frame_features_a, timestamps_a, frame_features_b, timestamps_b


def assert_tmk_model(model):
    d = 512
    n_frames_a = 100
    n_frames_b = 150
    offset = 20
    p = len(model.tmk.T)

    xa, tsa, xb, tsb = fake_data(d, n_frames_a, n_frames_b, offset)
    tmk_fv_a = model.single_fv(xa, tsa)
    tmk_fv_b = model.single_fv(xb, tsb)
    assert tmk_fv_a.shape == (1, p, 2 * model.tmk.m, d)
    assert tmk_fv_b.shape == (1, p, 2 * model.tmk.m, d)

    offsets = torch.arange(-n_frames_b, n_frames_b).view(1, -1).float()
    scores = model.score_pair(tmk_fv_a, tmk_fv_b, offsets)
    assert scores.shape == (1, 2 * n_frames_b)

    found_offset_idx = torch.argmax(scores).item()
    found_offset = offsets[0][found_offset_idx]
    tol = 2  # expect to find the right offset, with some tolerance
    assert -offset - tol <= found_offset < -offset + tol


def test_tmk_poullot():
    model = TMK_Poullot(ModelArgs(m=16))
    assert_tmk_model(model)


def test_tmk():
    model = TMK(ModelArgs(m=16, norm="freq", T=[2731, 4391, 9767, 14653]))
    assert_tmk_model(model)


def test_cte():
    m = 64
    d = 512
    n_frames_a = 100
    n_frames_b = 150
    offset = 20
    model = CTE(ModelArgs(m=m))
    xa, tsa, xb, tsb = fake_data(d, n_frames_a, n_frames_b, offset)

    tmk_fv_a = model.single_fv(xa, tsa)
    tmk_fv_b = model.single_fv(xb, tsb)

    assert tmk_fv_a.shape == (1, d, 2 * m)
    assert tmk_fv_b.shape == (1, d, 2 * m)

    offsets = torch.arange(-n_frames_b, n_frames_b).view(1, -1).float()
    scores = model.score_pair(tmk_fv_a, tmk_fv_b, offsets=offsets)
    assert scores.shape == (1, 2 * n_frames_b)

    found_offset_idx = torch.argmax(scores).item()
    found_offset = offsets[0][found_offset_idx]
    tol = 2  # expect to find the right offset, with some tolerance
    assert -offset - tol <= found_offset < -offset + tol
