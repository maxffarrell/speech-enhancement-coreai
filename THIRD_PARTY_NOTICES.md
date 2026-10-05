# Third-party notices

**Original model credits are prominently listed in [ATTRIBUTION.md](ATTRIBUTION.md). The Core AI converter is not the original model author.**

The root MIT license covers new conversion and validation code only.

- UL-UNAS implementation, streaming implementation, and DNS3 checkpoint: Xiaobin Rong and contributors, https://github.com/Xiaobin-Rong/ul-unas, revision `00f7c700da43d38347f30a6ccebd86fcbc798e07`. Repository MIT license reproduced at `vendor/ulunas/LICENSE`. Streaming implementation is credited upstream to Kailai Shen. `stream.py` is extracted before the demo block; unused ONNX imports were removed. The functional adapter clones input caches; architecture and pretrained weights are retained.
- RNNoise implementation, sparsification helpers, and official pretrained checkpoints: Jean-Marc Valin and RNNoise contributors, https://github.com/xiph/rnnoise, revision `70f1d256acd4b34a572f999a05c87bf00b67730d`. Repository BSD-3-Clause-style COPYING reproduced at `vendor/rnnoise/LICENSE`; individual files also carry BSD-style notices, retained verbatim. Checkpoints came from the exact official archive identified by upstream `model_version` and `download_model.sh`; its SHA-256 was verified. No training or dataset redistribution is performed.
- Converted weights retain their respective upstream license and attribution. Exporting does not create a new license for pretrained parameters. Upstream repositories do not supply separate checkpoint license files; distribution follows their repository licenses, with all corresponding notices included.
- Apple's Core AI libraries are dependencies, not redistributed here; their own license and platform terms apply.

UL-UNAS paper: Rong et al., “UL-UNAS: Ultra-Lightweight U-Nets for Real-Time Speech Enhancement via Network Architecture Search,” IEEE TASLP 34, 1085–1096 (2026), https://doi.org/10.1109/TASLPRO.2026.3661271.
RNNoise: Jean-Marc Valin, “A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement,” https://arxiv.org/abs/1709.08243. These checkpoints use the current architecture; they are not the original 2017 network.
