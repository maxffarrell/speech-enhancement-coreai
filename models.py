"""Pretrained model loaders and explicit streaming boundaries. MIT; see vendor licenses."""

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "ulunas"))
sys.path.insert(0, str(ROOT / "vendor" / "rnnoise"))
from rnnoise import RNNoise
from stream import StreamULUNAS
from ulunas import ULUNAS


class FunctionalULUNAS(StreamULUNAS):
    def forward(self, mix, conv_cache, tfa_cache, inter_cache):
        return super().forward(
            mix, conv_cache.clone(), tfa_cache.clone(), inter_cache.clone()
        )


class StreamingRNNoise(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(
        self, features, conv1_cache, conv2_cache, gru1_state, gru2_state, gru3_state
    ):
        m = self.source
        c1 = torch.cat((conv1_cache.clone(), features.transpose(1, 2)), dim=2)
        x = torch.tanh(m.conv1(c1))
        c2 = torch.cat((conv2_cache.clone(), x), dim=2)
        x = torch.tanh(m.conv2(c2)).transpose(1, 2)
        y1, h1 = m.gru1(x, gru1_state.clone())
        y2, h2 = m.gru2(y1, gru2_state.clone())
        y3, h3 = m.gru3(y2, gru3_state.clone())
        cat = torch.cat((x, y1, y2, y3), dim=-1)
        return (
            torch.sigmoid(m.dense_out(cat)),
            torch.sigmoid(m.vad_dense(cat)),
            c1[:, :, 1:],
            c2[:, :, 1:],
            h1,
            h2,
            h3,
        )


def load_model(variant, dtype=torch.float16):
    if variant == "ulunas_dns3":
        checkpoint = ROOT / "weights" / "ulunas_dns3.tar"
        source = ULUNAS().eval()
        source.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        )
        model = FunctionalULUNAS().eval()
        model.load_state_dict(source.state_dict(), strict=True)
        args = (torch.randn(1, 257, 1, 2), *model.init_caches())
        ins = ["mix", "conv_cache", "tfa_cache", "inter_cache"]
        outs = ["enh", "conv_cache_out", "tfa_cache_out", "inter_cache_out"]
    else:
        checkpoint = ROOT / "weights" / f"{variant}.pth"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        source = RNNoise(**saved["model_kwargs"]).eval()
        source.load_state_dict(saved["state_dict"], strict=True)
        model = StreamingRNNoise(source).eval()
        args = (
            torch.randn(1, 1, source.input_dim),
            torch.zeros(1, source.input_dim, 2),
            torch.zeros(1, source.cond_size, 2),
            *(torch.zeros(1, 1, source.gru_size) for _ in range(3)),
        )
        ins = [
            "features",
            "conv1_cache",
            "conv2_cache",
            "gru1_state",
            "gru2_state",
            "gru3_state",
        ]
        outs = ["gains", "vad", *[f"{x}_out" for x in ins[1:]]]
    return (
        model.to(dtype),
        tuple(x.to(dtype) for x in args),
        ins,
        outs,
        checkpoint,
        source.to(dtype),
    )
