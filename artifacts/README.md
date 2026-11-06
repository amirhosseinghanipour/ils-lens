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

`wikitext-training-frequencies.pt` contains complete WikiText-103 training token-ID
counts for the local Qwen tokenizer, including zero-frequency model vocabulary
rows. Its JSON companion records pinned dataset/tokenizer hashes, count totals,
measured preparation costs, and the source commit. These counts support the
required corpus-frequency mask; unseen tokens are not declared globally useless.

`qwen-primary-baselines.json` verifies the complete 32-prompt J/R fitting bank:
all layers, fp32 finite matrices, analytic final identities, source provenance,
file hashes, and measured costs. Large per-prompt tensor checkpoints remain in
the local run directory. Fitted-baseline completion does not establish held-out
faithfulness or readout performance.

`replacement-scale-diagnostic.json` records a fixed scale sweep of the retained
untrained live-TopK counterexample. All nonzero scales fail finite derivative/effect
agreement; this is a tiny numerical diagnostic, not a trained-model causal claim.
It neither selects a production scale nor relaxes the graph validation criterion.
