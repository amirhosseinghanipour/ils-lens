# Toy research extension

These measurements concern the existing four-layer HMM transformer, not the
0.5B Qwen model. They do not establish generalization across model seeds.

## Upstream lens readout

The actual J-lens and J++ reference code was used for the new baselines. J/R and
J++ expert fitting used the same 128 HMM fitting sequences; J++ weights used 32
separate labeled dev sequences (seed 60000). The 40 evaluation sequences were
not used to choose those weights. ILS reuses its previously fitted maps.

Important differences: upstream maps sum over future targets, while ILS predicts
a normalized horizon. Upstream fitting uses every valid source position, while
the original ILS pilot sampled one source per fitting sequence. J++ excludes
the final block; other methods have four source layers. Its eight clusters use
PCA16 because the toy hidden size is 16. All toy symbols have alphanumeric names,
so readout filtering is a no-op. These differences prevent an unqualified
method-superiority interpretation.

| Lens | Available layers | Any-layer recall@1 [95% CI] | Any-layer recall@10 [95% CI] |
|---|---|---|---|
| logit | 0–3 | 0.000 [0.000, 0.000] | 0.725 [0.575, 0.850] |
| ILS | 0–3 | 0.000 [0.000, 0.000] | 0.975 [0.925, 1.000] |
| J-reference | 0–3 | 0.300 [0.175, 0.450] | 0.975 [0.925, 1.000] |
| R-reference | 0–3 | 0.000 [0.000, 0.000] | 0.975 [0.925, 1.000] |
| J++ (label-tuned) | 0–2 | 0.725 [0.575, 0.826] | 1.000 [1.000, 1.000] |

Per-layer results and fitted matrices are in `artifacts/toy-reference/`.
Intervals use 200 item bootstrap resamples. Recall@10 is permissive in a
14-token vocabulary. Degenerate intervals from all-success/all-failure samples
do not prove zero population uncertainty.

## Random-direction gate

For eight new prompts, STATE was appended after non-report position 40. The
target was the probability of the true state ID as the next token. Direction
norm was fixed at 0.1 times the clean residual norm. Each lens/site intervention
was compared with 100 equal-norm covariance-matched directions; a pass requires
a signed probability increase above their 95th percentile. There was no tuning
of scale or site on these evaluation prompts.

| Lens | Layer | Pass rate [95% CI] | Prompts |
|---|---|---|---|
| logit | 0 | 0.000 [0.000, 0.000] | 8 |
| logit | 1 | 0.000 [0.000, 0.000] | 8 |
| logit | 2 | 0.000 [0.000, 0.000] | 8 |
| ILS | 0 | 0.125 [0.000, 0.375] | 8 |
| ILS | 1 | 0.125 [0.000, 0.375] | 8 |
| ILS | 2 | 0.000 [0.000, 0.000] | 8 |
| J-reference | 0 | 0.125 [0.000, 0.375] | 8 |
| J-reference | 1 | 0.125 [0.000, 0.375] | 8 |
| J-reference | 2 | 0.000 [0.000, 0.000] | 8 |
| R-reference | 0 | 0.125 [0.000, 0.375] | 8 |
| R-reference | 1 | 0.250 [0.000, 0.625] | 8 |
| R-reference | 2 | 0.000 [0.000, 0.000] | 8 |
| J++ (label-tuned) | 0 | 0.000 [0.000, 0.000] | 8 |
| J++ (label-tuned) | 1 | 0.000 [0.000, 0.000] | 8 |
| J++ (label-tuned) | 2 | 0.000 [0.000, 0.000] | 8 |

Label-tuned J++ has strong toy state readout here but passes no tested causal
gate. ILS passes only two of 24 site tests. These small samples do not establish
between-lens differences in causal selectivity; intervals overlap. No gate pass
is claimed for untested prompts or sites. All random effects are saved.

## Transcoders and budget closure

Four TopK skip transcoders (8× width, k=8) were trained for 200 optimization steps
on 2048 distinct HMM tokens each. This is far below the requested Qwen training
budget and is a synthetic validation experiment only. Explained variance is
measured on eight independent evaluation sequences. Dead fractions mean no
activation in this evaluation corpus, not permanent feature death.

| Layer | Variance explained [95% CI] | Evaluation-dead fraction |
|---|---|---|
| 0 | 0.707 [0.637, 0.745] | 0.602 |
| 1 | 0.616 [0.447, 0.715] | 0.508 |
| 2 | 0.792 [0.766, 0.813] | 0.539 |
| 3 | 0.823 [0.794, 0.844] | 0.461 |

Clean CE: 1.4795; all-transcoder CE: 1.7467; all-MLP-ablated CE: 2.4019. CE recovery: 0.710 [0.641, 0.754].

The graph freezes clean attention probabilities, normalization inverse factors,
and independent feature-source activations. The ILS read is a linear graph
target, never a graph node. Completeness is the share of absolute direct target
attribution from feature and embedding sources; errors and the affine bias
remainder count as unexplained. This is a declared local graph decomposition,
not proof that the unfrozen network has the same causal structure.

Per-item lens coverage uses 32 independent held-out donor sequences, epsilon
.05 and gamma=1. Donors use position 40 here, a deviation from the main random
donor-position protocol. Thresholds for low coverage and low completeness were
both fixed at .5 before the run. Unexplained coverage is 1−R², un-clipped.

| Item | Block | Lens R² | 1−R² | Graph completeness | Dark matter | Frozen edge relative error | Norm-only effect / predicted |
|---|---|---|---|---|---|---|---|
| 0 | 0 | -3.215 | 4.215 | 0.940 | False | 2.1e-07 | 1.000 |
| 0 | 1 | -1.267 | 2.267 | 0.928 | False | 2.2e-07 | 0.951 |
| 0 | 2 | -23.507 | 24.507 | 0.933 | False | 1.1e-07 | 1.053 |
| 0 | 3 | -15.697 | 16.697 | 0.929 | False | 6e-08 | 1.003 |
| 1 | 0 | -2.685 | 3.685 | 0.944 | False | 1.9e-07 | 1.000 |
| 1 | 1 | -1.038 | 2.038 | 0.930 | False | 1.9e-07 | 1.013 |
| 1 | 2 | -10.369 | 11.369 | 0.923 | False | 2.3e-07 | 0.911 |
| 1 | 3 | -12.932 | 13.932 | 0.917 | False | 1.5e-07 | 0.882 |
| 2 | 0 | -9.005 | 10.005 | 0.959 | False | 1.3e-07 | 1.000 |
| 2 | 1 | -2.797 | 3.797 | 0.950 | False | 2e-07 | 0.915 |
| 2 | 2 | -10.485 | 11.485 | 0.945 | False | 7e-08 | 0.854 |
| 2 | 3 | -15.091 | 16.091 | 0.938 | False | 0 | 0.857 |
| 3 | 0 | -2.544 | 3.544 | 0.937 | False | 2.9e-07 | 1.000 |
| 3 | 1 | -0.674 | 1.674 | 0.923 | False | 0 | 0.993 |
| 3 | 2 | -3.623 | 4.623 | 0.914 | False | 1.4e-07 | 0.941 |
| 3 | 3 | -9.944 | 10.944 | 0.906 | False | 4.8e-07 | 1.096 |
| 4 | 0 | -4.460 | 5.460 | 0.920 | False | 1.2e-07 | 1.000 |
| 4 | 1 | -1.926 | 2.926 | 0.913 | False | 1.9e-07 | 0.969 |
| 4 | 2 | -12.500 | 13.500 | 0.929 | False | 3.4e-07 | 0.958 |
| 4 | 3 | -13.895 | 14.895 | 0.917 | False | 2e-07 | 1.099 |
| 5 | 0 | -4.293 | 5.293 | 0.974 | False | 2e-07 | 1.000 |
| 5 | 1 | -1.816 | 2.816 | 0.937 | False | 1.7e-07 | 0.938 |
| 5 | 2 | -10.033 | 11.033 | 0.938 | False | 2.2e-07 | 1.006 |
| 5 | 3 | -38.357 | 39.357 | 0.920 | False | 3.3e-07 | 0.985 |
| 6 | 0 | -6.425 | 7.425 | 0.945 | False | 7.4e-07 | 1.000 |
| 6 | 1 | -2.358 | 3.358 | 0.913 | False | 1.2e-07 | 0.994 |
| 6 | 2 | -15.616 | 16.616 | 0.914 | False | 0 | 0.999 |
| 6 | 3 | -15.492 | 16.492 | 0.918 | False | 6e-08 | 0.959 |
| 7 | 0 | -9.054 | 10.054 | 0.959 | False | 2.1e-07 | 1.000 |
| 7 | 1 | -3.513 | 4.513 | 0.941 | False | 0 | 0.975 |
| 7 | 2 | -18.583 | 19.583 | 0.941 | False | 0 | 0.948 |
| 7 | 3 | -34.664 | 35.664 | 0.929 | False | 1.7e-07 | 1.048 |

All 32 checked edges pass: True. Dark-matter fraction: 0.000 [0.000, 0.000].

Dark-matter case list: none under these declared thresholds in these 32 toy
item/layer cases. That is not a conclusion about Qwen or an estimate of all
model computation. The bootstrap interval is degenerate because no sampled
case qualified. Norm-only survival still holds feature sources fixed; it does
not represent a completely unfrozen intervention. Transcoder reconstruction
quality and graph completeness are different quantities.

## Compute and provenance

| Experiment | Pre-run estimate | Actual seconds |
|---|---|---|
| Reference toy lenses | 180 s | 13.11 |
| Toy random gate | 180 s | 19.35 |
| Toy budget closure | 180 s | 3.35 |

All experiments use fp32 CPU computation with two threads. Configs, seeds,
matrices, transcoders and per-item outcomes are under the corresponding
`artifacts/toy-*` directories. The graph run was repeated with unchanged data
and settings to add prompt-bootstrap CE/variance intervals; no evaluation-based
model or hyperparameter changes were made. An upstream attribute-name mismatch
was corrected before the reference readout evaluation completed.
