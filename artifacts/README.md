# Included pilot artifacts

`toy/` contains a synthetic four-layer HMM transformer checkpoint, per-layer
fitting/held-out intervention tensors, fitted local J/R/ILS maps and CV metadata,
and a JSON results table. `qwen-pilot.json` records a numerical diagnostic on
the local Qwen model; no Qwen weights are included. `plots/` contains figures
generated from those JSON results.

Use `torch.load(path, map_location='cpu', weights_only=True)` for tensor bundles.
Only load artifacts from sources you trust. Model construction and experiment
settings are in `ilslens/eval/toy_hmm.py` and `configs/toy.yaml`.

The included results are from a single trained seed and are preliminary.
See `../RESULTS.md` for limitations and the correction to CV prior estimation.
`SHA256SUMS` records checksums of the included data files.

`qwen-numerical-validation.json` contains the completed 24-layer, three-epsilon
normalized-J equivalence validation. `numerical-all-layer-cosine-audit.json`
records hashes and stable reporting reductions of all saved matrix pairs. These
are numerical validation results, not the primary donor-intervention study.
The fp32 directional-error limitation remains documented in `REPORT.md`.
