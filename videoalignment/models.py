# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn

from .circulant_temporal_encoding import CirculantTemporalEncoding
from .temporal_match_kernel import TemporalMatchKernel


class Model(nn.Module):
    def __init__(self, args):
        self.args = args
        super(Model, self).__init__()

    def single_fv(self, ts, xs):
        raise NotImplementedError

    def shift_fv(self, fv, offset):
        raise NotImplementedError

    def score_pair(self, fv_a, fv_b, offsets=None):
        raise NotImplementedError

    def forward_pair(self, ts_a, ts_b, xs_a, xs_b, offsets=None):
        if offsets is None:
            length = ts_a.size()[1]
            offsets = torch.arange(-length, length).unsqueeze(0)
            offsets = offsets.to(ts_a.device)
        fv_a = self.single_fv(ts_a, xs_a)
        fv_b = self.single_fv(ts_b, xs_b)
        return self.score_pair(fv_a, fv_b, offsets)

    def forward(self, *args, **kwargs):
        return self.forward_pair(*args, **kwargs)

    def score(self, *args, **kwargs):
        return self.score_pair(*args, **kwargs)


class SumAggregation(Model):
    def __init__(self, args):
        super(SumAggregation, self).__init__(args)

    def single_fv(self, ts, xs):
        fv = torch.sum(ts, 1)
        fv = fv / (torch.sqrt(torch.sum(fv ** 2, 1, keepdim=True) + 10e-8))
        return fv

    def score_pair(self, fv_a, fv_b, offsets=None):
        return torch.sum(fv_a * fv_b, 1).view(-1, 1)


class CTE(Model):
    def __init__(self, args):
        super(CTE, self).__init__(args)
        self.cte = CirculantTemporalEncoding(m=self.args.m, lmbda=0.1)

    def single_fv(self, ts, xs):
        return self.cte.single_fv(ts.data)

    def score_pair(self, fv_a, fv_b, offsets=None, max_len=None):
        return self.cte.merge(fv_a.data, fv_b.data, offsets, max_len)


class TMK_Poullot(Model):
    def __init__(self, args):
        super(TMK_Poullot, self).__init__(args)
        self.tmk = TemporalMatchKernel(
            [2731, 4391, 9767, 14653],
            m=self.args.m,
            init="von_mises",
            normalize_l1=False,
            normalization="freq",
        )
        for p in self.tmk.parameters():
            p.requires_grad = False

    def single_fv(self, ts, xs):
        fv = self.tmk.single_fv(ts, xs)
        return fv

    def shift_fv(self, fv, offset):
        return self.tmk.shift_fv(fv, offset)

    def score_pair(self, fv_a, fv_b, offsets=None):
        return self.tmk.merge(fv_a, fv_b, offsets)


class TMK(Model):
    def __init__(self, args):
        super(TMK, self).__init__(args)
        self.tmk = TemporalMatchKernel(
            self.args.T,
            m=self.args.m,
            init="von_mises",
            normalize_l1=False,
            normalization=self.args.norm,
        )
        for p in self.tmk.parameters():
            p.requires_grad = False

        if self.args.use_pca:
            self.mean = torch.load(self.args.pca_mean)
            self.DVt = torch.load(self.args.pca_DVt)

    def single_fv(self, ts, xs):
        if self.args.use_pca:
            b_s, T, d = ts.size()
            ts = ts.view(-1, d)
            ts = ts - self.mean.expand_as(ts)
            ts = torch.mm(self.DVt, ts.transpose(0, 1)).transpose(0, 1)
            ts = ts / torch.sqrt(torch.sum(ts ** 2, dim=1, keepdim=True))
            ts = ts.view(b_s, T, d)

        fv = self.tmk.single_fv(ts, xs)
        return fv

    def shift_fv(self, fv, offset):
        return self.tmk.shift_fv(fv, offset)

    def score_pair(self, fv_a, fv_b, offsets=None):
        return self.tmk.merge(fv_a, fv_b, offsets)
