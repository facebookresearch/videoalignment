# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
from torch import nn


class CirculantTemporalEncoding(nn.Module):
    def __init__(self, m, lmbda):
        super(CirculantTemporalEncoding, self).__init__()
        self.m = m
        self.lmbda = lmbda

    def single_fv(self, ts):
        device = ts.device
        b, length, d = ts.size()
        # pad with 0 up to the closest power of 2
        length_power_2 = 2 ** int(np.ceil(np.log2(length)))
        offset = length_power_2 - length
        if offset > 0:
            zero_pad = torch.zeros(b, offset, d, dtype=torch.float32, device=device)
            ts = torch.cat([ts, zero_pad], dim=1)
        ts_ar = ts.permute(0, 2, 1).contiguous()  # shape (b, d, length)
        ts_ai = torch.zeros(ts_ar.size()).to(device)  # imaginary part is zero

        # With pytorch.fft, real and im are in the same tensor
        ts_air = torch.stack((ts_ar, ts_ai), dim=-1)
        Q = torch.fft(ts_air, 1)
        Qir = Q[..., 0]
        Qii = Q[..., 1]

        Qir = Qir[:, :, : self.m]
        Qii = Qii[:, :, : self.m]
        return torch.cat([Qir, Qii], dim=-1)

    def merge(self, fv_a, fv_b, offsets, max_len):
        max_len = offsets.shape[-1] // 2
        device = fv_a.device
        ts_ar = fv_a[:, :, : self.m]
        ts_ai = fv_a[:, :, self.m :]
        ts_br = fv_b[:, :, : self.m]
        ts_bi = fv_b[:, :, self.m :]
        length_power_2 = 2 ** int(np.ceil(np.log2(max_len)))

        Rrs = []
        # s(x, y) contains the scores for y shifted from 0 to max_len-1
        # compute s(a, b) and s(b, a), then concat flip(s(a, b)) and s(b, a) to have all possible time-shifts
        for i, (ts_ar, ts_ai, ts_br, ts_bi) in enumerate(
            [(ts_ar, ts_ai, ts_br, ts_bi), (ts_br, ts_bi, ts_ar, ts_ai)]
        ):
            Qir, Qii = ts_ar, ts_ai
            Bir, Bii = ts_br, ts_bi
            # See equation (10) in the paper
            Qdenr = torch.sum(Qir * Qir + Qii * Qii, dim=1, keepdim=True) + self.lmbda

            Sr = torch.sum((Qir * Bir + Qii * Bii) / Qdenr, dim=1, keepdim=True)
            Si = torch.sum((-Qii * Bir + Qir * Bii) / Qdenr, dim=1, keepdim=True)

            # padding with 0 up to the closest power of 2
            s0, s1, s2 = Sr.size()
            padding = length_power_2 - s2
            if padding > 0:
                zero_pad = torch.zeros(
                    s0, s1, padding, dtype=torch.float32, device=device
                )
                Sr = torch.cat([Sr, zero_pad], dim=2)
                Si = torch.cat([Si, zero_pad], dim=2)

            # With pytorch.fft, real and im are in the same tensor
            Sir = torch.stack((Sr, Si), dim=-1)
            R = torch.ifft(Sir, 1)
            Rr = R[..., 0]
            # Shape [b, 1, length_power_2], remove dim=1 and keep only up to max_len -> new shape [b, max_len]
            Rr = Rr.permute(0, 2, 1).squeeze(2)[:, :max_len]
            Rrs.append(Rr)

        Rrs[0] = Rrs[0].flip(dims=[1])
        outs = torch.cat(Rrs, dim=1)
        return outs

    def forward(self, ts_a, ts_b, offsets):
        fv_a = self.single_fv(ts_a)
        fv_b = self.single_fv(ts_b)
        len_a = ts_a.size()[1]
        len_b = ts_b.size()[2]
        max_len = max(len_a, len_b)
        return self.merge(fv_a, fv_b, offsets, max_len)
