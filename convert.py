"""Reproduce Core AI export and validate every output over independent state trajectories."""

import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("USE_OS_COREAI", "1")
from coreai.runtime import AIModel, AIModelAssetMetadata, NDArray, SpecializationOptions
from coreai_torch import TorchConverter, get_decomp_table
from models import ROOT, load_model


def psnr(ref, actual):
    a, b = np.asarray(ref, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 300.0
    return float(10 * np.log10(max(np.max(np.abs(a)), 1e-12) ** 2 / mse))


def export(variant, dtype):
    model, args, ins, outs, _checkpoint, _ = load_model(variant, dtype)
    ep = torch.export.export(model, args=args, strict=False).run_decompositions(
        get_decomp_table()
    )
    program = (
        TorchConverter()
        .add_exported_program(ep, input_names=ins, output_names=outs)
        .to_coreai()
    )
    path = ROOT / "exports" / f"{variant}_{str(dtype).split('.')[-1]}_streaming.aimodel"
    if path.exists():
        raise FileExistsError(path)
    meta = AIModelAssetMetadata()
    meta.author = (
        "Xiaobin Rong and contributors"
        if variant == "ulunas_dns3"
        else "Jean-Marc Valin and RNNoise contributors"
    )
    meta.license = (
        "MIT"
        if variant == "ulunas_dns3"
        else "BSD-3-Clause; see vendor/rnnoise/LICENSE and source-file notices"
    )
    meta.model_description = f"{variant} pretrained streaming neural network; Core AI conversion by Max Farrell. Audio DSP is external. See README and provenance.json."
    program.save_asset(path, meta)
    return path


async def validate(variant, dtype, path, count=32, wav=None):
    model, args, ins, outs, _, source = load_model(variant, dtype)
    runtime = await AIModel.load(path, SpecializationOptions.cpu_only())
    fn = runtime.load_function("main")
    assert list(fn.desc.input_names) == ins
    assert list(fn.desc.output_names) == outs
    ts, cs = args[1:], args[1:]
    gen = torch.Generator().manual_seed(1234)
    frames = torch.randn(
        (1, 257, count, 2) if variant == "ulunas_dns3" else (1, count, 65),
        generator=gen,
    ).to(dtype)
    if variant == "ulunas_dns3":
        import soundfile as sf

        if wav:
            audio, sr = sf.read(wav, dtype="float32")
        else:
            sr = 16000
            t = np.arange(16000, dtype=np.float32) / sr
            audio = (
                0.15 * np.sin(2 * np.pi * 180 * t)
                + 0.05 * np.random.default_rng(1234).standard_normal(16000)
            ).astype(np.float32)
        assert sr == 16000 and audio.ndim == 1
        z = torch.stft(
            torch.tensor(audio)[None],
            512,
            256,
            512,
            torch.hann_window(512),
            return_complex=True,
        )
        frames = torch.view_as_real(z)[:, :, :count].to(dtype)
        with torch.no_grad():
            spec = frames.permute(0, 3, 2, 1)
            feat = source.erb.bm(
                torch.log10(torch.norm(spec, dim=1, keepdim=True).clamp(1e-12))
            )
            feat, skips = source.encoder(feat)
            mask = source.erb.bs(source.decoder(source.dpgrnn(feat), skips))
            offline = (spec * mask).permute(0, 3, 2, 1)
    scores = {name: [] for name in outs}
    streaming = []
    ref_scores = []
    with torch.no_grad():
        for i in range(count):
            x = (
                frames[:, :, i : i + 1, :]
                if variant == "ulunas_dns3"
                else frames[:, i : i + 1, :]
            )
            pre_state = ts
            ref = model(x, *ts)
            runtime_out = await fn(
                {
                    name: NDArray(data=t.contiguous())
                    for name, t in zip(ins, (x, *cs), strict=True)
                }
            )
            values = [runtime_out[name].numpy() for name in outs]
            ts = tuple(t.detach() for t in ref[-len(ts) :])
            cs = tuple(torch.from_numpy(t.copy()) for t in values[-len(cs) :])
            for name, r, v in zip(outs, ref, values, strict=True):
                assert np.isfinite(v).all(), name
                scores[name].append(psnr(r.numpy(), v))
            if variant == "ulunas_dns3":
                streaming.append(ref[0])
            elif i >= 4:
                direct = source(frames[:, i - 4 : i + 1, :], list(pre_state[2:]))
                ref_scores.append(
                    min(
                        psnr(direct[0].numpy(), ref[0].numpy()),
                        psnr(direct[1].numpy(), ref[1].numpy()),
                    )
                )
    if streaming:
        ref_scores = [psnr(offline.numpy(), torch.cat(streaming, dim=2).numpy())]
    result = {
        "variant": variant,
        "precision": str(dtype),
        "device": "CPU",
        "frames": count,
        "input_fixture": str(wav) if wav else "deterministic synthetic input",
        "minimum_psnr_db": {k: min(v) for k, v in scores.items()},
        "source_graph_psnr_db": min(ref_scores),
    }
    print(json.dumps(result, indent=2))
    assert min(ref_scores) >= 70, "Streaming wrapper differs from upstream graph"
    assert min(min(v) for v in scores.values()) >= 40, "Core AI parity failed"
    (
        ROOT / "docs" / f"{variant}_{str(dtype).split('.')[-1]}_validation.json"
    ).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "variant", choices=["ulunas_dns3", "rnnoise10Ga_12", "rnnoise10Gb_15"]
    )
    p.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument(
        "--wav", type=Path, help="Optional 16 kHz mono WAV for UL-UNAS parity"
    )
    a = p.parse_args()
    dtype = getattr(torch, a.dtype)
    path = (
        ROOT / "exports" / f"{a.variant}_{a.dtype}_streaming.aimodel"
        if a.validate_only
        else export(a.variant, dtype)
    )
    asyncio.run(validate(a.variant, dtype, path, wav=a.wav))
