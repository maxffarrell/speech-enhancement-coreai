# Speech enhancement models for Apple Core AI

Pretrained **RNNoise** and **UL-UNAS** streaming neural networks exported as self-contained `.aimodel` directories. Conversion and validation by Max Farrell; original models and weights by the upstream authors linked below. This is an independent community conversion project.

| Model | Checkpoint | Asset | License |
| --- | --- | --- | --- |
| RNNoise | official `rnnoise10Ga_12.pth` | `exports/rnnoise10Ga_12_float32_streaming.aimodel` | upstream BSD-style notices |
| RNNoise | official `rnnoise10Gb_15.pth` | `exports/rnnoise10Gb_15_float32_streaming.aimodel` | upstream BSD-style notices |
| UL-UNAS | DNS3 `model_trained_on_dns3.tar` | `exports/ulunas_dns3_float32_streaming.aimodel` | MIT |

Download complete model directories from this repository, the [GitHub release](https://github.com/maxffarrell/speech-enhancement-coreai/releases/tag/v0.1.0), or the dedicated model repositories on Hugging Face: [RNNoise Core AI](https://huggingface.co/maxffarrell/rnnoise-coreai) and [UL-UNAS Core AI](https://huggingface.co/maxffarrell/ul-unas-coreai). Integrity hashes are in `SHA256SUMS`; upstream revisions and checkpoint hashes are in `provenance.json`.

## Reproduce

An Apple Silicon Mac with the Core AI runtime is required to run parity checks. Validated on macOS 27.0.1 (26A434), with CPU specialization. Export uses PyTorch 2.11.0, coreai-torch 0.4.3 and coreai-core 1.0.0b3. GPU, Neural Engine, iOS, and earlier OS releases have **not** been validated. No accelerator latency or energy-efficiency claim is made.

```sh
git clone https://github.com/maxffarrell/speech-enhancement-coreai.git
cd speech-enhancement-coreai
uv sync --frozen
# Check the published assets:
uv run python convert.py rnnoise10Ga_12 --validate-only
uv run python convert.py rnnoise10Gb_15 --validate-only
uv run python convert.py ulunas_dns3 --validate-only
# Optional real-audio check, using your own 16 kHz mono file:
uv run python convert.py ulunas_dns3 --validate-only --wav noisy.wav
# To regenerate, move the matching exports/*.aimodel directory elsewhere first:
uv run python convert.py ulunas_dns3 --dtype float32
```

The exporter refuses to overwrite existing assets. It loads checkpoint dictionaries using `weights_only=True` and strict state-dict matching, exports with `torch.export.export`, runs Apple's decomposition table, then saves the Core AI AIProgram with provenance/license metadata. No retraining, checkpoint edits, quantization, or third-party Metal kernels are involved.

## UL-UNAS audio integration

[Upstream code, checkpoints and paper](https://github.com/Xiaobin-Rong/ul-unas). The checkpoint is trained on DNS3. The upstream frame-streaming implementation is used, with input caches cloned at the export boundary to preserve functional state semantics.

Audio is **16 kHz mono**, with a 512-point periodic Hann-window STFT, 256-sample hop (16 ms), centered reflect padding as in upstream PyTorch, and matching inverse STFT. The exported graph accepts one real/imaginary STFT frame, rather than raw audio.

| Input | float32 shape | Output |
| --- | --- | --- |
| `mix` | `[1,257,1,2]` | `enh` (same shape) |
| `conv_cache` | `[1,5358]` | `conv_cache_out` |
| `tfa_cache` | `[1,402]` | `tfa_cache_out` |
| `inter_cache` | `[1,1056]` | `inter_cache_out` |

Initialize caches to zero per stream. Feed each returned cache into the matching input for the next frame. Reset all caches when starting a new stream. Centered STFT requires buffering; the 16 ms hop is not a claim of zero-lookahead end-to-end latency.

```sh
uv run python enhance.py noisy.wav enhanced.wav
```

This complete WAV example runs the neural network in Core AI on CPU; PyTorch is used only for STFT/ISTFT. It preserves upstream handling of the trailing audio samples. It does not resample or downmix automatically.

## RNNoise integration

[Upstream RNNoise](https://github.com/xiph/rnnoise) ([canonical repository](https://gitlab.xiph.org/xiph/rnnoise)). Both checkpoints come from the exact official model archive in `provenance.json`; its SHA-256 matches upstream `model_version`. They use the current 65-feature, 32-gain architecture, with 128 convolution conditioning channels and 384 recurrent units. No claim is made that either checkpoint exactly matches upstream's quantized default C model.

RNNoise operates on **48 kHz mono**, 480-sample (10 ms) audio frames. Its native DSP produces **65 normalized features**. The Core AI graph converts these to **32 band gains** and a voice-activity score. Native pitch analysis/filtering, feature extraction, gain smoothing/application, and waveform synthesis remain external. This asset alone does not accept PCM or generate cleaned audio. Preserve upstream feature conventions and DSP when integrating; do not pass raw PCM, GTCRN features, or arbitrary STFT bins as `features`.

| Input | float32 shape | Output |
| --- | --- | --- |
| `features` | `[1,1,65]` | `gains` `[1,1,32]`, `vad` `[1,1,1]` |
| `conv1_cache` | `[1,65,2]` | `conv1_cache_out` |
| `conv2_cache` | `[1,128,2]` | `conv2_cache_out` |
| `gru1_state` | `[1,1,384]` | `gru1_state_out` |
| `gru2_state` | `[1,1,384]` | `gru2_state_out` |
| `gru3_state` | `[1,1,384]` | `gru3_state_out` |

Initialize all five states to zero and carry all outputs forward on every frame. The convolution caches make the valid-convolution training graph causal for frame-wise inference. Cold-start caches match the native streaming convention. The source check compares against upstream PyTorch sliding five-frame windows after the four-frame warmup, with identical incoming GRU states.

## Validation and limits

Every output is compared over 32 consecutive frames with independent PyTorch and Core AI state trajectories, including convolution caches and GRU states. Reports are in `docs/`.

- RNNoise 10Ga float32: minimum 104.44 dB across all outputs; source graph agreement 119.08 dB.
- RNNoise 10Gb float32: minimum 108.11 dB; source graph agreement 115.74 dB.
- UL-UNAS float32 on upstream real-audio frames: minimum 127.83 dB; streaming versus upstream offline spectral graph 160.58 dB. A separate deterministic synthetic-input report is also included.
- Full 10-second UL-UNAS WAV example versus upstream waveform output: 144.11 dB PSNR, 160,000 output samples, all finite.
- RNNoise float16 was rejected: VAD minimum PSNR was 35.13 dB, below the 40 dB gate. UL-UNAS upstream float16 export hits a recurrent input/weight dtype mismatch. Only validated float32 assets are distributed.

These are conversion-parity checks, not speech-quality benchmark scores. RNNoise testing uses synthetic normalized-feature-shaped inputs and does not establish parity with native quantized C/DSP, end-to-end audio quality, or latency. UL-UNAS real-audio testing uses the upstream `audio/noisy/0174.wav` sample locally; that recording is not redistributed. A conversion can reproduce upstream outputs without establishing quality on your microphone or noise environment.

## Licensing and attribution

New conversion/example code: [MIT](LICENSE). UL-UNAS source and weights retain [upstream MIT](vendor/ulunas/LICENSE). RNNoise source and weights retain [upstream BSD-style license](vendor/rnnoise/LICENSE) and individual source-file notices. The root license does not replace those terms. See [third-party notices](THIRD_PARTY_NOTICES.md) for authors, exact revisions, modifications, and citations. Model weights use the upstream repository licenses; no separate checkpoint-specific license was supplied upstream.

Related: [GTCRN Core AI](https://github.com/maxffarrell/gtcrn-coreai), including DNS3 and VCTK streaming assets. GTCRN is maintained separately and is not redistributed by this project.
