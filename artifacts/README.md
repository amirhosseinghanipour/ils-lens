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

`primary-samples-integrity-snapshot.json` records hashes and integrity checks for
the first 81 completed primary intervention shards. Collection was still running;
this is explicitly an incomplete snapshot, not a fitted or held-out result.

`primary-training-complete-integrity-snapshot.json` extends that audit to all
384 training shards and the first 25 held-out shards. All checked integrity
conditions pass; held-out collection and lens fitting were incomplete at this
snapshot. The earlier 81-shard snapshot is retained.

`primary-full-collection-integrity.json` verifies all 1,152 primary intervention
shards (384 training, 768 held-out). All checks pass and previously audited file
hashes are unchanged. This completes the collection-integrity audit, not lens
fitting, faithfulness evaluation or the full research study.

`qwen-baseline-norms.json` and `plots/qwen-baseline-norms.png` summarize the
completed 32-prompt J/R matrix bank. Norm intervals resample fitting prompts;
the norm-of-mean ratios describe cancellation, not causal faithfulness or a
rotation/curvature decomposition. Saved fp32 matrices are unchanged.
