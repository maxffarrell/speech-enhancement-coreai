# Copyright (c) 2026 Xiaobin Rong and contributors. MIT license.
# Extracted from upstream ulunas_onnx/stream/ulunas_stream.py; demo-only imports removed.
import os
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from tqdm import tqdm

from ulunas import ULUNAS


class StreamULUNAS(nn.Module):
    """Frame-wise streaming ULUNAS for ONNX export."""
    CONV_CACHE_SHAPES = [
        (1, 2, 129),
        (24, 1, 65),
        (24, 1, 33),
        (24, 1, 33),
        (12, 1, 33),
        (12, 2, 65),
    ]
    TFA_CACHE_HIDDEN = [24, 48, 48, 64, 32, 64, 48, 48, 24, 2]
    INTER_CACHE_SHAPES = [(33, 16), (33, 16)]

    def __init__(self):
        super().__init__()
        base = ULUNAS()
        self.erb = base.erb
        self.encoder = base.encoder
        self.dpgrnn = base.dpgrnn
        self.decoder = base.decoder

    @classmethod
    def init_caches(cls, batch_size=1, device=None):
        conv_size = sum(int(np.prod(shape)) for shape in cls.CONV_CACHE_SHAPES)
        tfa_size = sum(cls.TFA_CACHE_HIDDEN)
        inter_size = sum(int(np.prod(shape)) for shape in cls.INTER_CACHE_SHAPES)
        conv_cache = torch.zeros(batch_size, conv_size, device=device)
        tfa_cache = torch.zeros(batch_size, tfa_size, device=device)
        inter_cache = torch.zeros(batch_size, inter_size, device=device)
        return conv_cache, tfa_cache, inter_cache

    @classmethod
    def _unpack_conv_cache(cls, conv_cache):
        bsz = conv_cache.shape[0]
        caches = []
        offset = 0
        for shape in cls.CONV_CACHE_SHAPES:
            n = int(np.prod(shape))
            caches.append(conv_cache[:, offset : offset + n].view(bsz, *shape))
            offset += n
        return caches

    @classmethod
    def _pack_conv_cache(cls, caches):
        return torch.cat([cache.reshape(cache.shape[0], -1) for cache in caches], dim=1)

    @classmethod
    def _unpack_tfa_cache(cls, tfa_cache):
        bsz = tfa_cache.shape[0]
        caches = []
        offset = 0
        for hidden in cls.TFA_CACHE_HIDDEN:
            caches.append(tfa_cache[:, offset : offset + hidden].view(1, bsz, hidden))
            offset += hidden
        return caches

    @classmethod
    def _pack_tfa_cache(cls, caches):
        return torch.cat([cache.reshape(cache.shape[1], -1) for cache in caches], dim=1)

    @classmethod
    def _unpack_inter_cache(cls, inter_cache):
        bsz = inter_cache.shape[0]
        caches = []
        offset = 0
        for shape in cls.INTER_CACHE_SHAPES:
            n = int(np.prod(shape))
            caches.append(inter_cache[:, offset : offset + n].view(bsz, *shape))
            offset += n
        return caches

    @classmethod
    def _pack_inter_cache(cls, caches):
        return torch.cat([cache.reshape(cache.shape[0], -1) for cache in caches], dim=1)

    @staticmethod
    def _stream_temporal_conv(x, conv, cache):
        inp = torch.cat([cache, x], dim=2)
        if isinstance(conv, nn.ConvTranspose2d):
            # Match offline path: ZeroPad2d([0, 0, kt-1, 0]) + ConvTranspose2d, then
            # keep only the current-step output frame.
            kt = conv.kernel_size[0]
            inp_padded = torch.nn.functional.pad(inp, (0, 0, kt - 1, 0))
            y = conv(inp_padded)[:, :, -1:, :]
        else:
            y = conv(inp)
        new_cache = inp[:, :, 1:, :]
        return y, new_cache

    @staticmethod
    def _stream_ctfa(ctfa, x, h_cache):
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at, h_cache = ctfa.ta_gru(zt.transpose(1, 2), h_cache)
        at = torch.sigmoid(ctfa.ta_fc(at).transpose(1, 2))

        af = torch.sigmoid(ctfa.fa(x))
        y = at[..., None] * x * af[:, None]
        return y, h_cache

    def _stream_xconv(self, block, x, conv_cache, tfa_cache):
        x, conv_cache = self._stream_temporal_conv(x, block.ops[1], conv_cache)
        x = block.ops[2](x)
        x = block.ops[3](x)
        x, tfa_cache = self._stream_ctfa(block.ops[4], x, tfa_cache)
        x = block.ops[5](x)
        return x, conv_cache, tfa_cache

    def _stream_xdws(self, block, x, tfa_cache, conv_cache=None):
        h = block.pconv(x)
        if conv_cache is None:
            h = block.dconv[0](h)
            h = block.dconv[1](h)
        else:
            h, conv_cache = self._stream_temporal_conv(h, block.dconv[1], conv_cache)
        h = block.dconv[2](h)
        h = block.dconv[3](h)
        h, tfa_cache = self._stream_ctfa(block.dconv[4], h, tfa_cache)
        return h, conv_cache, tfa_cache

    def _stream_xmb(self, block, x, tfa_cache, conv_cache=None):
        residual = x
        x = block.pconv1(x)
        if conv_cache is None:
            x = block.dconv(x)
        else:
            x, conv_cache = self._stream_temporal_conv(x, block.dconv[1], conv_cache)
            x = block.dconv[2](x)
            x = block.dconv[3](x)

        x = block.pconv2[0](x)
        x = block.pconv2[1](x)
        x, tfa_cache = self._stream_ctfa(block.pconv2[2], x, tfa_cache)

        if x.shape == residual.shape:
            x = x + residual
        x = block.shuffle(x)
        return x, conv_cache, tfa_cache

    @staticmethod
    def _stream_dpgrnn(block, x, inter_cache):
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)

        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        intra_x = block.intra_rnn(intra_x)[0]
        intra_x = block.intra_fc(intra_x)
        intra_x = intra_x.reshape(x.shape[0], -1, block.width, block.input_size)
        intra_x = block.intra_ln(intra_x)
        intra_out = x + intra_x

        x = intra_out.permute(0, 2, 1, 3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x, inter_cache = block.inter_rnn(inter_x, inter_cache)
        inter_x = block.inter_fc(inter_x)
        inter_x = inter_x.reshape(x.shape[0], block.width, -1, block.input_size)
        inter_x = inter_x.permute(0, 2, 1, 3)  # (B,T,F,C)
        inter_x = block.inter_ln(inter_x)
        inter_out = intra_out + inter_x

        dual_out = inter_out.permute(0, 3, 1, 2)  # (B,C,T,F)
        return dual_out, inter_cache

    def forward(
        self,
        spec,
        conv_cache,
        tfa_cache,
        inter_cache,
    ):
        """
        spec: (B,F,T,2), streaming T=1
        """
        spec_ref = spec
        spec = spec.permute(0, 3, 2, 1)  # (B,2,T,F)
        feat = torch.log10(torch.norm(spec, dim=1, keepdim=True).clamp(1e-12))
        feat = self.erb.bm(feat)  # (B,1,T,129)

        (
            conv_cache_e0,
            conv_cache_e1,
            conv_cache_e2,
            conv_cache_d2,
            conv_cache_d3,
            conv_cache_d4,
        ) = self._unpack_conv_cache(conv_cache)
        (
            tfa_cache_0,
            tfa_cache_1,
            tfa_cache_2,
            tfa_cache_3,
            tfa_cache_4,
            tfa_cache_5,
            tfa_cache_6,
            tfa_cache_7,
            tfa_cache_8,
            tfa_cache_9,
        ) = self._unpack_tfa_cache(tfa_cache)
        inter_cache_0, inter_cache_1 = self._unpack_inter_cache(inter_cache)

        en_outs = []
        feat, conv_cache_e0, tfa_cache_0 = self._stream_xconv(
            self.encoder.en_convs[0], feat, conv_cache_e0, tfa_cache_0
        )
        en_outs.append(feat)

        feat, conv_cache_e1, tfa_cache_1 = self._stream_xmb(
            self.encoder.en_convs[1], feat, tfa_cache_1, conv_cache_e1
        )
        en_outs.append(feat)

        feat, conv_cache_e2, tfa_cache_2 = self._stream_xdws(
            self.encoder.en_convs[2], feat, tfa_cache_2, conv_cache_e2
        )
        en_outs.append(feat)

        feat, _, tfa_cache_3 = self._stream_xmb(self.encoder.en_convs[3], feat, tfa_cache_3)
        en_outs.append(feat)

        feat, _, tfa_cache_4 = self._stream_xdws(self.encoder.en_convs[4], feat, tfa_cache_4)
        en_outs.append(feat)

        feat, inter_cache_0 = self._stream_dpgrnn(self.dpgrnn[0], feat, inter_cache_0)
        feat, inter_cache_1 = self._stream_dpgrnn(self.dpgrnn[1], feat, inter_cache_1)

        feat, _, tfa_cache_5 = self._stream_xdws(
            self.decoder.de_convs[0], feat + en_outs[4], tfa_cache_5
        )
        feat, _, tfa_cache_6 = self._stream_xmb(
            self.decoder.de_convs[1], feat + en_outs[3], tfa_cache_6
        )
        feat, conv_cache_d2, tfa_cache_7 = self._stream_xdws(
            self.decoder.de_convs[2], feat + en_outs[2], tfa_cache_7, conv_cache_d2
        )
        feat, conv_cache_d3, tfa_cache_8 = self._stream_xmb(
            self.decoder.de_convs[3], feat + en_outs[1], tfa_cache_8, conv_cache_d3
        )
        feat, conv_cache_d4, tfa_cache_9 = self._stream_xconv(
            self.decoder.de_convs[4], feat + en_outs[0], conv_cache_d4, tfa_cache_9
        )
        m_feat = torch.sigmoid(feat)

        m = self.erb.bs(m_feat)
        spec_enh = spec * m
        spec_enh = spec_enh.permute(0, 3, 2, 1)  # (B,F,T,2)

        conv_cache = self._pack_conv_cache(
            [conv_cache_e0, conv_cache_e1, conv_cache_e2, conv_cache_d2, conv_cache_d3, conv_cache_d4]
        )
        tfa_cache = self._pack_tfa_cache(
            [
                tfa_cache_0,
                tfa_cache_1,
                tfa_cache_2,
                tfa_cache_3,
                tfa_cache_4,
                tfa_cache_5,
                tfa_cache_6,
                tfa_cache_7,
                tfa_cache_8,
                tfa_cache_9,
            ]
        )
        inter_cache = self._pack_inter_cache([inter_cache_0, inter_cache_1])

        return spec_enh, conv_cache, tfa_cache, inter_cache


