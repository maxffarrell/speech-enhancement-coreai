"""Enhance 16 kHz mono audio with the exported UL-UNAS neural network."""

import argparse
import asyncio
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("USE_OS_COREAI", "1")
from coreai.runtime import AIModel, NDArray, SpecializationOptions


async def enhance(asset, infile, outfile):
    audio, sr = sf.read(infile, dtype="float32")
    if sr != 16000 or audio.ndim != 1:
        raise ValueError("Expected 16 kHz mono WAV")
    window = torch.hann_window(512)
    spec = torch.view_as_real(
        torch.stft(
            torch.tensor(audio)[None], 512, 256, 512, window, return_complex=True
        )
    )
    model = await AIModel.load(asset, SpecializationOptions.cpu_only())
    fn = model.load_function("main")
    state = {
        "conv_cache": np.zeros((1, 5358), np.float32),
        "tfa_cache": np.zeros((1, 402), np.float32),
        "inter_cache": np.zeros((1, 1056), np.float32),
    }
    frames = []
    for i in range(spec.shape[2]):
        output = await fn(
            {
                "mix": NDArray(data=spec[:, :, i : i + 1].contiguous()),
                **{k: NDArray(data=v) for k, v in state.items()},
            }
        )
        frames.append(output["enh"].numpy().copy())
        state = {k: output[k + "_out"].numpy().copy() for k in state}
    enhanced = torch.view_as_complex(
        torch.from_numpy(np.concatenate(frames, axis=2)).contiguous()
    )
    wav = torch.istft(enhanced, 512, 256, 512, window).numpy()[0]
    wav = np.pad(wav, (0, len(audio) - len(wav)))
    sf.write(outfile, wav, 16000, subtype="FLOAT")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--asset",
        type=Path,
        default=Path(__file__).parent / "exports/ulunas_dns3_float32_streaming.aimodel",
    )
    a = p.parse_args()
    asyncio.run(enhance(a.asset, a.input, a.output))
