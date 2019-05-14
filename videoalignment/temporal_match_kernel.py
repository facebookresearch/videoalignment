# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
from scipy.special import iv as bessel
from torch import nn

from .utils import get_device


def project_to_probability_simplex(v):
    device = v.device
    nn, n = v.size()
    mu = torch.sort(v, dim=1, descending=True)[0].float()
    ns = torch.arange(1, n + 1).view(1, -1).float()
    ns = ns.to(device)
    cumsum = torch.cumsum(mu, dim=1).float() - 1
    arg = mu - cumsum / ns

    idx = [i for i in range(arg.size()[1] - 1, -1, -1)]
    idx = torch.LongTensor(idx)
    idx = idx.to(device)
    arg = arg.index_select(1, idx)

    rho = n - 1 - torch.max(arg > 0, dim=1)[1].view(-1, 1)
    theta = 1 / (rho.float() + 1) * cumsum.gather(1, rho)
    out = torch.max(v - theta, 0 * v).clamp_(10e-8, 1)
    return out


def tmk(ts, xs, a, ms, Ts):
    block_size = 500
    ts = ts.unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, len_ts, d)
    a_xp = torch.cat([a, a], dim=1).unsqueeze(2).unsqueeze(0)  # (1, T, 2m, 1)
    ms = (ms.unsqueeze(0) / Ts.unsqueeze(1)).unsqueeze(0).unsqueeze(3)  # (1, Ts, m, 1)

    for t in range(0, xs.size()[1], block_size):
        args = ms * xs[:, t : t + block_size].unsqueeze(1).unsqueeze(
            1
        )  # (b_s, Ts, m, len_ts)
        sin_cos = a_xp * torch.cat(
            [torch.sin(args), torch.cos(args)], dim=2
        )  # (b_s, Ts, 2m, len_ts)
        sin_cos = sin_cos.unsqueeze(4)  # (b_s, Ts, 2m, len_ts, 1)
        this_fv = torch.sum(
            sin_cos * ts[:, :, :, t : t + block_size], dim=3
        )  # (b_s, Ts, 2m, d)
        if t == 0:
            fv = this_fv
        else:
            fv += this_fv

    return fv


class TemporalMatchKernel(nn.Module):
    def __init__(
        self, T, normalization, beta=32, m=32, init="von_mises", normalize_l1=False
    ):
        """
        Temporal Match Kernel Layer
        :param T: Periods (list)
        :param beta: beta param of the modified Bessel function of the first kind
        :param m: number of Fourier coefficients per period
        :param init: type of initialization ('von_mises' or 'uniform': random from uniform distribution)
        :param normalize_l1: Whether to L1 normalize Fourier coefficents before each forward pass
        """
        super(TemporalMatchKernel, self).__init__()
        self.T = T
        self.beta = beta
        self.m = m
        self.normalize_l1 = normalize_l1
        self.normalization = normalization

        # Initialization
        if init == "von_mises":
            np_a = [
                (bessel(0, self.beta) - np.exp(-self.beta)) / (2 * np.sinh(self.beta))
            ] + [bessel(i, self.beta) / np.sinh(self.beta) for i in range(1, self.m)]
            np_a = np.asarray(np_a).reshape(1, -1)
            np_a = np.repeat(np_a, len(T), 0)
        elif init == "uniform":
            np_a = np.random.uniform(0, 1, (len(T), self.m))
        else:
            raise NotImplementedError

        self.a = nn.Parameter(torch.from_numpy(np_a).float())  # (T, m)
        self.ms = (2 * np.pi * torch.arange(0, self.m)).float()
        self.Ts = torch.tensor(self.T, dtype=torch.float32, requires_grad=False)

    def single_fv(self, ts, xs):
        device = get_device(self)
        self.ms = self.ms.to(device)
        self.Ts = self.Ts.to(device)
        self.a.data.clamp_(min=10e-8)

        if ts.dim() == 3:
            return tmk(ts, xs, torch.sqrt(self.a), self.ms, self.Ts)
        elif ts.dim() == 4:
            out = []
            for i in range(self.Ts.shape[0]):
                outi = tmk(
                    ts[:, :, i],
                    xs,
                    torch.sqrt(self.a[i : i + 1]),
                    self.ms,
                    self.Ts[i : i + 1],
                )
                out.append(outi)
            return torch.cat(out, 1)

    def merge(self, fv_a, fv_b, offsets):
        device = get_device(self)
        eps = 1e-8
        if "feat" in self.normalization:
            a_xp = self.a.unsqueeze(0).unsqueeze(-1)
            a_xp = torch.cat([a_xp, a_xp], dim=2)
            fv_a_0 = fv_a / torch.sqrt(a_xp)
            fv_b_0 = fv_b / torch.sqrt(a_xp)
            norm_a = torch.sqrt(torch.sum(fv_a_0 ** 2, dim=3, keepdim=True) + eps) + eps
            norm_b = torch.sqrt(torch.sum(fv_b_0 ** 2, dim=3, keepdim=True) + eps) + eps
            fv_a = fv_a / norm_a
            fv_b = fv_b / norm_b

        if "freq" in self.normalization:
            norm_a = (
                torch.sqrt(torch.sum(fv_a ** 2, dim=2, keepdim=True) / self.m + eps)
                + eps
            )
            norm_b = (
                torch.sqrt(torch.sum(fv_b ** 2, dim=2, keepdim=True) / self.m + eps)
                + eps
            )
            fv_a = fv_a / norm_a
            fv_b = fv_b / norm_b

        elif self.normalization == "matrix":
            norm_a = (
                torch.sqrt(
                    torch.sum(torch.sum(fv_a ** 2, dim=-1, keepdim=True), dim=2) + eps
                )
                + eps
            )  # (b_s, T, 1)
            norm_b = (
                torch.sqrt(
                    torch.sum(torch.sum(fv_b ** 2, dim=-1, keepdim=True), dim=2) + eps
                )
                + eps
            )  # (b_s, T, 1)

        fv_a_sin = fv_a[:, :, : self.m]  # (b_s, T, m, d)
        fv_a_cos = fv_a[:, :, self.m :]  # (b_s, T, m, d)
        fv_b_sin = fv_b[:, :, : self.m]  # (b_s, T, m, d)
        fv_b_cos = fv_b[:, :, self.m :]  # (b_s, T, m, d)

        self.ms = self.ms.to(device)

        xs = offsets.float()
        ms = self.ms.unsqueeze(1)  # (m, 1)

        dot_sin_sin = torch.sum(
            fv_a_sin * fv_b_sin, dim=3, keepdim=True
        )  # (b_s, T, m, 1)
        dot_sin_cos = torch.sum(
            fv_a_sin * fv_b_cos, dim=3, keepdim=True
        )  # (b_s, T, m, 1)
        dot_cos_cos = torch.sum(
            fv_a_cos * fv_b_cos, dim=3, keepdim=True
        )  # (b_s, T, m, 1)
        dot_cos_sin = torch.sum(
            fv_a_cos * fv_b_sin, dim=3, keepdim=True
        )  # (b_s, T, m, 1)

        T = torch.tensor(self.T, dtype=torch.float32, requires_grad=False)
        T = T.to(device)
        T = T.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        cos_delta = torch.cos(
            ms.unsqueeze(0).unsqueeze(0) * xs.unsqueeze(1).unsqueeze(1) / T
        )  # (b_s, T, m, delta)
        sin_delta = torch.sin(
            ms.unsqueeze(0).unsqueeze(0) * xs.unsqueeze(1).unsqueeze(1) / T
        )  # (b_s, T, m, delta)

        dots = (
            dot_sin_sin * cos_delta
            + dot_sin_cos * sin_delta
            + dot_cos_cos * cos_delta
            - dot_cos_sin * sin_delta
        )  # (b_s, T, m, delta)
        dots = torch.sum(dots, dim=2)  # (b_s, T, delta)

        if self.normalization == "matrix":
            dots = dots / (norm_a * norm_b)
        elif self.normalization == "freq":
            dots = dots / self.m
        elif self.normalization in ["feat", "feat_freq"]:
            dots = dots / 512
        dots = torch.mean(dots, dim=1)
        return dots

    def shift_fv(self, fv, offset):
        device = get_device(self)
        fv_sin = fv[:, :, : self.m]  # (b_s, T, m, d)
        fv_cos = fv[:, :, self.m :]  # (b_s, T, m, d)

        ms = self.ms.unsqueeze(1)  # (m, 1)
        T = torch.tensor(self.T, dtype=torch.float32, requires_grad=False)
        T = T.unsqueeze(0).unsqueeze(2).unsqueeze(2)

        self.ms = self.ms.to(device)
        T = T.to(device)

        sin_delta = torch.sin(
            ms.unsqueeze(0).unsqueeze(0) * offset / T
        )  # (b_s, T, m, 1)
        cos_delta = torch.cos(
            ms.unsqueeze(0).unsqueeze(0) * offset / T
        )  # (b_s, T, m, 1)

        fv_sin_shifted = fv_sin * cos_delta + fv_cos * sin_delta
        fv_cos_shifted = fv_cos * cos_delta - fv_sin * sin_delta
        fv_shifted = torch.cat([fv_sin_shifted, fv_cos_shifted], 2)
        return fv_shifted

    def forward(self, ts_a, ts_b, xs_a, xs_b, offsets):
        """
        Computes the TMK scores over two batch of sequences with the same length
        :param ts_a: First time series (b_s, length, d)
        :param ts_b: Second time series (b_s, length, d)
        :param xs_a: Timestamps of first series (b_s, length)
        :param xs_b: Timestamps of second series (b_s, length)
        :param offsets: Offsets for which the kernel score is computed (b_s, n_offsets)
        :return: Kernel scores for every temporal offset (b_s, 2*length)
        """
        fv_a = self.single_fv(ts_a, xs_a)
        fv_b = self.single_fv(ts_b, xs_b)
        return self.merge(fv_a, fv_b, offsets)
