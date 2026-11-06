# ILS-lens research report — incomplete Qwen study

Updated 2026-11-06. **The requested research criteria are not yet met.** Completed
measurements are the Qwen numerical/reference benchmarks and the trained toy HMM
experiments below. Production Qwen experiment runners are implemented but their
presence is not a result. Long CPU runs were stopped at the owner's request;
the all-layer GPU matrix-equivalence validation and primary J/R fitting have
completed. Training and held-out donor-intervention collection are complete;
primary GPU fitting is running. Missing measurements are
unknown, not null findings.

Full production execution is disabled during the specification audit. Independent
numerical validation and primary fitting run separately on the GPU. The current implementation is
not an exact implementation of the entire proposal.

The label-tuned J++ baseline has a separate queued controller following the
capability filter. It requires a matching successful predecessor and retained
dev items, then waits for 9.5 GiB free GPU memory. No retained dev items means
J++ is unavailable, not a negative readout result. Test labels never unlock
this stage. The runner retains eight clusters, 64 PCA dimensions and the same
32 fitting sequences as J/R; a tokenizer-only provenance check confirmed exact
row/hash agreement with the primary manifest. The existing unbenchmarked
estimate is 14,400 seconds and 1,500 MB output; actual cost remains unmeasured.
This queues baseline fitting only, not the disabled full study. Queue guards
passed within the 149-test local suite (6.457 seconds).

## Completed primary collection and integrity audit

Calibration now has an explicit `configs/calibration.yaml`, preserving the
existing 100,000 test-split activations, 128-token windows, skip-16 and
final-position exclusion, epsilon 0.05 and gamma 0.9. CLI overrides are recorded
in the resolved configuration and newly fitted calibration metadata. Changed
sampling conventions and fewer than 100,000 references are rejected. This is a
reproducibility correction, not a calibration measurement or method change.
The local suite passed 145 tests in 6.448 seconds after this change.

A [synthetic tail-cost probe](artifacts/calibration-tail-cost-probe.json) fitted
eight standard-normal score tails with 100,000 fp32 references each in 0.111
seconds, excluding imports, against a 30-second estimate. Median scalar-fit
time was 0.01281 seconds. Applying this rate to 96 core lens/layer cells and
62,776 corpus-observed tokens gives about 21.45 hours for tail fitting alone.
This is an arithmetic capacity scenario, not a production forecast: real tails
and optimizer convergence differ, some tails are unavailable, and GPU scoring,
activation collection, serialization, J++ and the released extra baseline are
excluded. The activation cache requires 8,601,600,000 bytes on disk. No sampling
budget or tail-fitting method was reduced in response to this estimate.

Calibration resume validation now binds each cached fit to its reference
provenance, layer, transport matrix and corpus-frequency tensor, and checks
reference count, vocabulary length and mask shape. Previously, the runner
checked the calibration method and matrix but could reuse a stale dead-token
mask after frequency changes. Unverified older caches are rejected; no
production calibration artifact existed when this correction was made. The
statistical method is unchanged. The local suite passed 150 tests in 6.377
seconds, including changed-frequency and incomplete-cache regression cases.

The primary-results exporter previously checked file count and fingerprint but
not the full cell grid. A synthetic 24-file fixture with duplicate layer 0 and
missing layer 23 was incorrectly accepted by the previous exporter. It now
rejects duplicate/missing cells, mismatched conventions or fitting sample count,
missing baselines, and nonfinite statistics. Negative R² and explicit undefined
statistics remain preserved. The complete local suite passed 147 tests in
6.190 seconds. No production fit or scientific selection rule changed.

The [full collection audit](artifacts/primary-full-collection-integrity.json)
passes for all 1,152 checkpoints: 384 training shards and 768 held-out shards,
totaling 19,827,816,192 bytes. Each layer/epsilon has 7,168 fitting samples and
512 held-out samples. All 409 files in the preceding snapshot retain their
hashes. The checked final-layer relative error remains at most 5.9134704e-6
against a 1e-4 tolerance. These are deterministic data-integrity checks, not
confidence intervals or evidence of independent donor draws.

The full audit took 72.298 seconds against a 120-second estimate, excluding
imports, and produced 283,576 bytes against a 400,000-byte estimate. Reproduce
with `scripts/audit_primary_samples.py --run runs/qwen-proposal-v4 --output
artifacts/primary-full-collection-integrity.json`. Both earlier snapshots are
retained. The supervisor has moved to GPU fitting with the original prior,
penalty, estimator, epsilon and horizon grids. Collection completion does not
establish fitted-lens faithfulness, readout quality, causal sensitivity or
completion of the full study.

## Completed primary training collection

All 384 training checkpoints (16 prompts × 24 layers) are present. The
[training-complete integrity snapshot](artifacts/primary-training-complete-integrity-snapshot.json)
verified all of them and the first 25 completed held-out checkpoints, covering
18,546,766,798 bytes. Every training layer has 7,168 donor interventions at each
of the six registered epsilon values; four horizons reuse the same forward
effects. The audit checks finite fp32 tensors, shapes, provenance, valid source
and donor indices, epsilon scaling, and the analytic final-layer horizon response.
The maximum checked final-layer relative error was 5.9134704e-6, below the
registered 1e-4 tolerance. This deterministic integrity check has no sampling CI.

The read-only audit took 75.365 seconds against a 90-second estimate and produced
100,264 bytes against a 200,000-byte estimate; timing excludes Python imports.
It used `scripts/audit_primary_samples.py --run runs/qwen-proposal-v4 --output
artifacts/primary-training-complete-integrity-snapshot.json`. The prior 81-shard
snapshot is preserved. This new snapshot is still incomplete for held-out
collection (409 of 1,152 total checkpoints); it does not establish donor
independence, fitted-lens performance, or full-study completion. Primary ILS
fitting has not started at this milestone.

The independent capability filter can be queued after the transcoder benchmark
with `scripts/capability_after_benchmark.py`. It requires the matching benchmark
controller to terminate successfully, waits for 9.5 GiB free GPU memory, and runs
the registered five-task filter without a per-task limit. The existing fallback
bank, 10 samples at temperature 0.8, and 8/10 correctness requirement are unchanged.
The configuration's pre-run estimate is 3,600 seconds and 100 MB, still
unbenchmarked. This queue produces capability records only; it does not start
readout scoring or bypass the graph-compliance block. Queueing is not evidence
that any model passes the capability filter.

The subliminal runner's outcome accounting now distinguishes an unusable teacher
cohort from measured non-transmission. A tiny integration test reproduced the old
behavior: no matched number-only examples nevertheless yielded
`any_transmits=false`. Missing configurations now retain unknown outcomes; the
aggregate is false only when every configuration was evaluated and none passed,
true when at least one evaluated configuration passed, and otherwise null.
Partial and entirely unevaluable attempts cannot pass the completion audit.
The protocol fingerprint records this reporting correction. Teacher sampling,
training, three-seed testing and the p<0.01 criterion are unchanged. All 142 local
tests pass, including reference checks; no production subliminal result exists yet.

The completion audit previously accepted arbitrary explicit stage statuses
unless they appeared on a short exclusion list. Seven regression cases reproduced
false acceptance (`failed`, `partial`, `running`, `queued`, `not_requested`, an
unknown status, and null). Explicit status now must be `measured`; legacy runners
without a top-level status retain their separate substantive checks. Measured
zero effects remain valid findings. This repairs completion accounting, not the
research method, and does not establish completeness of legacy artifacts. All
140 local tests pass, including reference checks (6.925 seconds).

The independent all-layer transcoder throughput benchmark can be queued with
`scripts/benchmark_after_primary.py` after the primary GPU controller terminates
successfully. The queue refuses failed/incomplete or mismatched controller
records and checks for 9.5 GiB free GPU memory. It performs the registered three
warmup and ten timed updates, then discards its weights; it does not start 100M-token
training or enable the full study. Its pre-run estimate remains 600 seconds and
10 MB of output, pending measurement. The queue validation and complete local
suite pass (136 tests, 5.919 seconds, including reference checks).

The initial fp32 GPU suffix benchmark completed: six-epsilon antithetic batches
at source position 64 took 0.2033, 0.1279, 0.0659 and 0.000575 seconds at blocks
0, 7, 15 and 23 respectively, averaged over three repetitions. See
[the timing artifact](artifacts/qwen-gpu-suffix-benchmark.json). These timings
exclude data loading and do not establish scientific validity. The all-layer
isotropic-limit validation was launched separately with an unbenchmarked
24-hour / 3-GB estimate; its completed result is reported below.

## Descriptive norms of the completed J/R baseline bank

The 32 fitted per-prompt matrices were summarized without loading a model or
refitting any lens. Their original fp32 artifact hashes were checked; these
reporting reductions use float64. The table reports mean per-prompt Frobenius
norms with 95% descriptive bootstrap intervals (1,000 prompt resamples, seed
4101, paired across lenses/layers). Shared-article dependence is not modeled.
The ratio columns give norm of the mean matrix divided by mean matrix norm;
these describe cancellation in this fixed bank and have no inference interval.
All maps retain the upstream **summed-future-target** convention. Block 23 is the
analytic identity, so its identical norms are a construction check.

| Block | J mean norm [95% CI] | R mean norm [95% CI] | J ratio | R ratio | J/R mean-matrix cosine |
| --- | --- | --- | --- | --- | --- |
| 0 | 63.41 [60.56, 66.78] | 42.66 [41.37, 44.07] | 0.615 | 0.838 | 0.727 |
| 1 | 53.22 [50.84, 55.96] | 40.27 [39.05, 41.59] | 0.658 | 0.838 | 0.737 |
| 2 | 49.21 [46.85, 51.94] | 39.71 [38.44, 41.09] | 0.671 | 0.836 | 0.729 |
| 3 | 49.81 [47.41, 52.54] | 42.07 [40.64, 43.57] | 0.699 | 0.851 | 0.757 |
| 4 | 43.00 [41.52, 44.60] | 37.55 [36.57, 38.58] | 0.752 | 0.865 | 0.761 |
| 5 | 41.47 [40.30, 42.75] | 36.05 [35.27, 36.84] | 0.792 | 0.879 | 0.791 |
| 6 | 41.17 [40.02, 42.42] | 36.79 [36.02, 37.58] | 0.802 | 0.886 | 0.807 |
| 7 | 41.58 [40.53, 42.71] | 38.58 [37.79, 39.35] | 0.804 | 0.888 | 0.806 |
| 8 | 43.35 [42.31, 44.46] | 40.06 [39.27, 40.85] | 0.790 | 0.885 | 0.804 |
| 9 | 45.81 [44.70, 46.99] | 42.59 [41.74, 43.44] | 0.780 | 0.887 | 0.801 |
| 10 | 46.22 [45.15, 47.33] | 41.54 [40.76, 42.29] | 0.790 | 0.899 | 0.818 |
| 11 | 50.98 [49.73, 52.28] | 45.12 [44.25, 45.94] | 0.806 | 0.908 | 0.830 |
| 12 | 45.71 [44.82, 46.66] | 40.91 [40.23, 41.57] | 0.818 | 0.914 | 0.838 |
| 13 | 46.09 [45.26, 46.97] | 41.37 [40.73, 42.02] | 0.832 | 0.923 | 0.851 |
| 14 | 43.44 [42.69, 44.28] | 39.22 [38.68, 39.78] | 0.855 | 0.940 | 0.857 |
| 15 | 47.04 [46.33, 47.84] | 40.90 [40.42, 41.42] | 0.888 | 0.954 | 0.891 |
| 16 | 46.24 [45.71, 46.80] | 41.15 [40.74, 41.55] | 0.906 | 0.962 | 0.909 |
| 17 | 43.56 [43.10, 44.02] | 38.75 [38.41, 39.06] | 0.918 | 0.968 | 0.927 |
| 18 | 41.79 [41.44, 42.15] | 36.73 [36.51, 36.96] | 0.930 | 0.973 | 0.933 |
| 19 | 41.41 [41.09, 41.75] | 35.44 [35.24, 35.65] | 0.941 | 0.975 | 0.945 |
| 20 | 38.10 [37.89, 38.34] | 32.67 [32.56, 32.79] | 0.962 | 0.983 | 0.957 |
| 21 | 37.35 [37.15, 37.57] | 32.53 [32.45, 32.63] | 0.978 | 0.991 | 0.965 |
| 22 | 34.23 [34.14, 34.33] | 31.30 [31.26, 31.33] | 0.992 | 0.997 | 0.978 |
| 23 | 29.93 [29.93, 29.93] | 29.93 [29.93, 29.93] | 1.000 | 1.000 | 1.000 |

![Baseline norm diagnostics](artifacts/plots/qwen-baseline-norms.png)

Early matrices show more cancellation under averaging in this bank. This does
not identify rotation versus curvature, demonstrate workspace structure, or
establish a faithfulness advantage for either lens. These norms are of averaged
per-prompt maps, not individual Jacobians, so they do not measure a gradient
explosion distribution. No model selection used this diagnostic.

Full values are in [qwen-baseline-norms.json](artifacts/qwen-baseline-norms.json).
The summary took 12.796 seconds versus a pre-run estimate of 60 seconds; its
66,353-byte JSON exceeded the 50,000-byte output estimate. The separate figure
is a rendering of those saved values, not an additional experiment.

## Primary sample integrity snapshot

A read-only snapshot audited 81 completed primary intervention shards (3.903 GB)
while collection continued. All inspected tensors were finite fp32 with matching
provenance, registered shapes, valid source/donor indices and consistent scaling
across epsilon. For the three available final-layer shards, the maximum
per-sample relative error against `delta * source_horizon_weight` was
**4.7085e-6**, below the audit tolerance of 1e-4. This is an identity check, not
held-out predictive performance. The snapshot contains only 81 of the expected
1,152 shards; completeness is explicitly false. Hashes and checks are preserved
in [primary-samples-integrity-snapshot.json](artifacts/primary-samples-integrity-snapshot.json).
The audit took 15.526 seconds without loading a model. All 137 local tests pass
(6.598 seconds, including reference checks and deliberate sample-corruption tests).

## Budget-closure and reporting validation

Two tiny planted-run integration cases (zero and three matched completion pairs)
exposed an abort before independent model-difference measurements. The runner now
retains WikiText model-difference measurements and any eligible behavior-context
measurements, but reports `contrast_status=insufficient_matched_prompts` and
`detection_status=partial`. It does not execute a contrast/null test below the
unchanged four-prompt minimum, resample completions, or report a negative
detection result. Cohort counts and partial status survive report export; the
completion audit still rejects missing contrast rows. The four-prompt measured
path remains covered. All 138 local tests pass (6.389 seconds, including reference
checks). No production planted result exists or was overwritten; the runner's
cohort protocol is recorded in its experiment fingerprint.

The downstream report exporter also failed two new regressions: it omitted the
explicit insufficient-cohort status/counts from contrast results and the
`exploratory_affine` flag from graph results. Generated Markdown and JSON now
retain the recorded outcome, contrast cohort counts and excluded-pair IDs.
The reporting CLI counts result artifacts rather than calling every existing
artifact a measured stage. All 135 tests pass with local reference checks enabled
(6.134 seconds under unittest). No experimental outcomes or thresholds changed.

An adversarial regression test exposed eight input-validation failures: nonfinite
R² or completeness values, and completeness outside [0,1], could escape validation
when the paired measurement was missing. Such values could silently enlarge the
unknown-case count instead of exposing a failed calculation. Budget closure now
rejects these invalid inputs independently. Explicit `null` measurements remain
unknown; finite negative R² remains measured, with unexplained variance `1 − R²`
allowed to exceed one. This correction changes no thresholds or valid measured
results. All 133 tests pass, including the local upstream layer-convention and
WorkspaceBench tokenizer checks (5.753 seconds under unittest).

## Ablation finite-precision ordering corrected before production

The primary collector subtracts antithetic branches per position before applying
the horizon kernel. The ablation collector instead subtracted horizon-reduced
branches; its one-sided variant also used different reduction paths for patched
and clean states. These expressions are equal in exact arithmetic but need not
be numerically equivalent in fp32.

A new tiny-model regression sets the final residual to a common value of 1,000,
patches one source by 0.01, and checks all four horizons against the directly
represented per-position change. Before correction, the Gaussian/isotropic
antithetic path had a maximum relative discrepancy of 1.524%; the one-sided path
had a maximum relative discrepancy of 10.506 (1,050.6%). These are synthetic
collection-arithmetic errors, not Qwen readout scores or sampling estimates.

The corrected ablation collector differences per position first. Gaussian and
isotropic variants use the same antithetic path as primary collection; one-sided
responses subtract the clean per-position state before averaging. All 131 tests
pass. Artifact metadata records `per-position-before-horizon-v2`; diagnostic
fitting rejects older or unverified ablation difference-order metadata. No
production ablation results were overwritten, no evaluation labels or epsilon
selection were involved, and the active immutable primary run is unchanged.
This correction does not remove the separately reported small-epsilon JVP error.

## Completed primary J/R baseline fitting

J and R were fitted on the same 32 shuffled WikiText training prompts used by the
primary source-plus-donor corpus, across all 24 block-output layers. The
[baseline artifact audit](artifacts/qwen-primary-baselines.json) verifies all
64 per-prompt checkpoints: 768 matrices per lens, all 896×896, fp32, finite, and
bound to the primary corpus fingerprint. Both lenses use the upstream summed
future-target convention and target the final block output before final norm.
The final-block matrix is the documented analytic identity, not an upstream fit.
No evaluation labels were used.

The `baselines-complete` event records 1,339.57 seconds (22.33 minutes), including
model/data preparation, against the pre-run 14,232-second estimate. Checkpoint
files occupy 4,933,026,496 bytes. The read-only artifact audit took 14.64 seconds;
it records individual file hashes, prompt provenance, and per-prompt fitting
times. [Reproduce the audit](scripts/audit_primary_baselines.py) after completion.
This is completion of J/R fitting only: no primary held-out R2, readout score,
causal pass rate, or J++ result is established by it. Donor-intervention collection
is now running in the same immutable execution checkout.

## Completed full training-vocabulary counts and corrected provenance check

The dead-row mask requires zero-frequency tokens from the complete training
corpus. The tokenizer-only count runner now binds the saved counts to the pinned
WikiText manifest, loaded split fingerprint, and tokenizer/config file hashes.
It preserves the existing tokenization rule (every row followed by a newline,
without added special tokens), includes unused model-vocabulary rows, and checks
all 1,801,350 training records. Existing counts are reused only after provenance,
shape, nonnegativity, record-count, and total-count checks pass.

The pre-run estimate is 7,200 seconds and 2 MB. This preparation uses two CPU
threads and loads no model weights; it does not move model fitting off the GPU.
A tiny counting test verifies exact results across flush sizes, empty rows,
newline counts, unused vocabulary rows, and out-of-vocabulary rejection. The
complete corpus count has now finished; the measured counts are below.

The first production attempt failed before token counting: it compared the loaded
Dataset fingerprint to the download manifest. The serialized `state.json`
fingerprint matches the manifest (`b06c41c32c07fc3b`), but the installed library's
`load_from_disk` calls `with_format`, yielding `87cfe39fb693eadb` in memory.
The corrected runner checks the serialized identity and record count, records
both identities, and hashes each Arrow file. This fixes a provenance-check bug;
it changes neither corpus contents nor tokenization. The
[failure record](artifacts/corpus-frequency-provenance-failure.json) is retained,
and a save/load regression test covers this case and rejects changed saved
identity or record count. The corrected count completed, and a verified reuse invocation left the saved artifact byte-for-byte unchanged.

The completed count covers **1,801,350 records and 124,881,340 Qwen tokenizer
tokens**, including the declared newline after each record. Of 151,936 model
vocabulary rows, 62,776 occur and **89,160 (58.6826%) have zero frequency** in this
training corpus. This is a census under the fixed tokenization rule, so no sampling
CI is appropriate. Corpus absence does not imply globally dead or meaningless
model rows. The prescribed mask will also include tiny unembedding-row norms;
readout top-10 mask rates remain unmeasured until the readout experiments run.

Actual runner time was 206.34 seconds versus 7,200 seconds estimated. The count
artifact occupies 1,218,827 bytes versus the 2-MB estimate. See the
[complete counts](artifacts/wikitext-training-frequencies.pt) and
[metadata, source hashes, and costs](artifacts/wikitext-training-frequencies.json).
The failed first attempt remains recorded separately; it produced no counts and
was not combined with this result. No GPU model computation was used for counting.

## Completed all-layer normalized-J equivalence validation

All 72 registered layer/epsilon cells pass matrix cosine ≥ 0.95 and mean held-out
top-10 overlap ≥ 0.8, including all 24 layers at the smallest epsilon, 0.0001.
The run uses eight fitting contexts and 20 held-out contexts, with d=896 rotated
orthogonal directions per fitting context (N=7,168 per layer/epsilon), gamma=1,
and lambda=0. Layers refer to block outputs and the target is pre-final-norm.

| epsilon | Passing layers | Minimum matrix cosine (reporting fp64) | Minimum held-out top-10 overlap | Maximum directional JVP error |
| --- | --- | --- | --- | --- |
| 0.01 | 24/24 | 0.999961956 | 0.975 | 1.371% |
| 0.001 | 24/24 | 0.999999314 | 0.990 | 0.991% |
| 0.0001 | 24/24 | 0.999933094 | 0.955 | 7.783% |

The [complete result and provenance](artifacts/qwen-numerical-validation.json)
retain the original fp32 cosine values. The table uses float64 *reporting
reductions* on the saved fp32 maps, documented in the
[all-layer cosine audit](artifacts/numerical-all-layer-cosine-audit.json); no model
or fitting computation was changed to float64. These are fixed-test extrema,
not estimated population effects. Per-prompt overlap values were not saved by
this run, so no overlap bootstrap CI is reconstructed or invented.

This validates convergence toward our own **normalized-horizon** averaged J on
the original prefix-sampled contexts. It does not compare primary donor ILS with
the upstream summed-target J/R baselines and does not establish readout quality,
causal faithfulness, or superiority. The final maximum directional JVP error
remains 7.783%, so the matrix pass does not remove the fp32 limitation below.

Measured experiment time was 9,582.61 seconds (2.66 hours), versus the pre-run
86,400-second estimate. The experiment recorded 4,779,235,290 bytes before its
final result/log writes, versus the 3,000-MB estimate. Final directory bytes are
also recorded in the artifact. This confirms the earlier storage underestimate.
The numerical process exited normally; the waiting supervisor verified all cells
and launched primary J/R fitting on the corrected shuffled corpus in the fixed
`cd400d0` execution checkout. J/R fitting has since completed and donor collection remains in progress, with
no primary faithfulness result yet.

Reproduce publication with `scripts/report_numerical_validation.py` after running
the read-only cosine audit over layers 0–23. The large numerical tensors remain
in the local run directory; the public audit records each matrix file's hash.

## Partial Qwen JVP check: fp32 still has a small-perturbation error floor

A read-only audit of all completed contexts at layers 0–17 records 432 saved
finite-difference/JVP comparisons: eight original fitting contexts, one direction
per context, and three epsilon values at each layer. The running experiment was
not changed. Relative error is the norm of the antithetic difference minus the
autograd JVP, divided by the JVP norm (floored at 1e-12).

| epsilon | Mean relative error | Descriptive 95% context interval | Maximum observed |
| --- | --- | --- | --- |
| 0.01 | 0.165% | [0.134%, 0.200%] | 1.371% |
| 0.001 | 0.269% | [0.165%, 0.373%] | 0.991% |
| 0.0001 | 2.484% | [1.603%, 3.429%] | 7.783% |

The worst case is layer 2, context 4 at epsilon 0.0001 (7.783%). The increase in
error at smaller epsilon is consistent with finite-precision cancellation, but
this audit does not isolate roundoff from all other numerical effects. Running in
fp32 does **not** justify assuming negligible directional finite-difference error.
A high averaged-matrix cosine can coexist with these discrepancies. No epsilon,
precision, threshold, or model method was changed in response to this snapshot.

Intervals resample the eight context identities jointly across layers, 1,000 times.
They describe only this fixed prefix-sampled context bank and do not account for
shared-article dependence. They are not population CIs over arbitrary prompts or
directions. The existing queued gate checks the registered matrix-cosine and
held-out top-10 criteria; passing it would not certify every directional JVP or
complete the research. At the time of this partial snapshot, full-layer results were pending; the completed
all-layer result is now reported above.

See the [raw errors and source hashes](artifacts/numerical-partial-jvp-audit.json),
[reproduction script](scripts/audit_numerical_jvp_errors.py), and
[plot script](scripts/plot_numerical_jvp_errors.py). The final read-only audit took
8.33 seconds; it loaded saved tensors on CPU and performed no model fitting.

![Partial fp32 directional errors](artifacts/plots/qwen-partial-jvp-errors.png)

## Primary-run resource projection before collection

The [read-only resource audit](scripts/audit_primary_resources.py) records a
[primary-resource projection](artifacts/primary-resource-projection.json) bound to
the registered configuration and measured suffix benchmark by SHA-256. It counts
26,606,174,208 bytes of tensors for the primary interventions, indices, per-prompt
J/R matrices, and 576 selected ILS matrices. Serialization metadata, temporary
atomic copies, cached means, and downstream experiments require additional space.
This is a final storage lower bound, not peak RAM or VRAM.

Piecewise-linear interpolation across the four measured GPU layer timings yields
17,680.34 seconds (4.91 hours) of suffix execution for 7,680 training-plus-held-out
sample draws at each of 24 layers. Six epsilons are batched, and all four horizons
share each antithetic forward pair; neither count is multiplied into time again.
The benchmark uses a single source position and prompt. This projection excludes
bank construction, transfers outside the timed suffix call, checkpoint writes,
J/R fitting, ILS cross-validation, and every downstream phase. It has no estimated
confidence interval and is not a complete-run duration forecast.

The offline fitter's old “conservative CPU” label was inaccurate for its CUDA
execution default. Current source labels the unchanged 180-seconds-per-cell value
as an unbenchmarked placeholder and records the execution device. The already
queued immutable checkout remains unchanged and retains the old label; this
report explicitly corrects its interpretation. No GPU CV throughput measurement
has yet been obtained, and no sample budget or method was reduced. The accounting
test compares its storage formula with actual tiny tensor layouts and verifies
that timing counts each batched forward pair only once.

## Matched graph comparisons preserve exact inputs and missing cases

Synthetic regression cases exposed silent overwriting of duplicate graph sites,
acceptance of changed or missing input identities, propagation of nonfinite target
values, and omission of unmeasured base/tuned pairs. Graph rows now record hashes
of the exact input token sequence and read-target vector. The comparison rejects
duplicate sites, unverified or changed contexts, changed measured target vectors,
and nonfinite measured target values. It retains unmeasured pairs with their base
and tuned statuses, a null difference, and explicit measured/unmeasured counts.
Only measured pairs enter the existing prompt-level bootstrap and control null.

The tiny graph runner also rejects a cached row when its target vector changes,
even if the primary manifest fingerprint is unchanged. Existing graph artifacts
without the new identities are not accepted for the updated comparison. These
checks do not establish complete artifact provenance or resolve the additional
independent-feature freeze: graph runs remain exploratory and production-blocked.
All 127 small-input tests passed at that correction; no new Qwen graph measurement is claimed.

## Insufficient contrast cohorts remain unmeasured

A tiny-model runner test with fixed generated-token fixtures reproduced two
previous crashes: zero eligible completion pairs failed state extraction, and
three eligible pairs failed the existing four-prompt minimum for split-half nulls.
The contrast runner now preserves all completions and excluded prompt indices and
writes `insufficient_matched_prompts` with no measured rows. It does not shorten
the registered offsets, generate replacement samples, lower the minimum, or
interpret the missing test as negative evidence. The artifact audit explicitly
rejects this status as an incomplete contrast measurement.

The regression test also checks the four-prompt boundary using actual tiny-model
state extraction and the 1,000-split null procedure for both on/off-policy modes
and both averaging orders. Sampling is a fixed fixture; this is implementation
validation, not a Qwen behavioral or readout result. The original synthetic failures
were observed before the correction, and the suite passed 123 tests at that correction.

## Primary-run routing correction

The serial study launcher previously relied on every consumer's legacy
`runs/qwen-main` default. The separately queued primary experiment uses a different
registered output directory. The launcher and final artifact audit now accept
`--primary-run`, and explicitly route all primary producers and consumers to it.
A regression test checks all 22 affected stages and confirms that the audit does
not read a legacy manifest when an explicit run is selected. This prevents mixing
runs; it changes no estimator or data and has not altered the active immutable
numerical run or queued primary checkout. Full production remains disabled.

## Transcoder corpus: overlap failure and independently audited correction

Both pinned FineWeb and UltraChat parquet shards passed their size and SHA-256
checks. Downloading them took 6654.7 seconds against the original
7,200-second estimate. A separate local reverification took 1.31 seconds;
that is not the download duration. The prepared JSONL contains 198,242 training
and 21,752 validation documents after document deduplication, totaling
780,757,466 bytes. Preparation itself took 4.64 seconds, excluding download and
the subsequent tokenizer audit.

[Verified shard metadata](artifacts/transcoder-shards-verified.json) and
[prepared corpus manifest](artifacts/transcoder-corpus-100m-manifest.json) record
pinned revisions and hashes. The [completed tokenizer audit](artifacts/transcoder-corpus-100m-budget.json)
took 171.06 seconds. It found all 100,000,000 distinct training tokens (703,125 web
and 78,125 chat windows), but **46 validation windows overlap the selected training
prefix**. The original corpus and failed audit are retained. No transcoder
training has started on this corpus.

The tested correction in `scripts/deduplicate_transcoder_validation.py` removes
entire validation documents containing overlapping windows into a separate corpus,
verifies that training records remain byte-for-byte identical, and independently
repeats the budget and overlap audit. Its pre-run estimate is 600 seconds and
780,757,466 additional bytes. Actual correction plus independent audit took
364.13 seconds and the corrected corpus occupies 780,596,212 bytes. The correction
removed 27 validation documents (21 web and 6 chat), leaving 21,725 validation
documents and all 198,242 training documents. Training lines are byte-for-byte
unchanged. The independent rescan **passes**: 100M distinct training tokens and
zero overlaps among 112,757 unique validation windows. This checks exact windows,
not all possible semantic or near-duplicate contamination.

The [corrected manifest](artifacts/transcoder-corpus-100m-clean-manifest.json) and
[corrected audit](artifacts/transcoder-corpus-100m-clean-budget.json) preserve hashes,
excluded document IDs, source commit, and costs. This correction responds to corpus
contamination without inspecting model outcomes or evaluation labels. The full
transcoder GPU benchmark and training remain unrun. The new tiny-input test checks
preservation of the original corpus and training rows, exclusion of the offending
validation document, and the independently recomputed zero-overlap budget.

## Numerical-run resource snapshot

At 2026-11-06 06:56 UTC, the numerical run had written 82/192 context checkpoints
and 30/72 matrix checkpoints after 6,551.6 seconds of wall time. Its files occupied
2,036,313,791 bytes. These are partial actual costs, not completion costs or GPU
kernel time. The original estimate remains 86,400 seconds and 3,000 MB.

The stored tensor layout gives a final lower bound of **4,778,360,832 bytes**:
192 contexts each store seven fp32 896×896 matrices, and 72 fitted comparisons
each store two. This exceeds the original disk estimate before serialization
metadata, logs and atomic-save temporary copies. Available local disk remains
sufficient; no sample count, precision or checkpoint retention was reduced.
See [the storage audit](artifacts/numerical-partial-storage-audit.json) and
[the read-only reproduction script](scripts/audit_numerical_storage.py).
This accounting applies only to numerical validation; the full study's compute
cost and completion time remain unmeasured.

## Partial GPU numerical checkpoint audit

A read-only snapshot of completed blocks 0–6 contains 21 isotropic-limit matrix
comparisons (8 contexts × 896 directions per cell). Float64 reductions of the
saved fp32 matrices give the following cosines against our normalized-horizon J:

| Block | ε=.01 | ε=.001 | ε=.0001 |
|---|---:|---:|---:|
| 0 | 0.99996196 | 0.99999982 | 0.99998251 |
| 1 | 0.99997944 | 0.99999980 | 0.99998069 |
| 2 | 0.99999506 | 0.99999971 | 0.99997084 |
| 3 | 0.99999888 | 0.99999957 | 0.99995824 |
| 4 | 0.99999919 | 0.99999951 | 0.99995194 |
| 5 | 0.99999925 | 0.99999950 | 0.99995151 |
| 6 | 0.99999969 | 0.99999944 | 0.99994484 |

These are numerical checks conditional on the fixed test inputs, not population
estimates or evidence of ILS superiority; no sampling CIs are claimed. The original
fp32 reductions occasionally exceed 1 by about 0.00002. Both reductions are
preserved in [the audit artifact](artifacts/numerical-partial-cosine-audit.json),
with input-file hashes and source commit. The matrices and running experiment
remain unchanged. The read-only audit took 0.23 seconds excluding Python startup.

This is **not** the completed equivalence test: held-out top-10 overlap is not
stored in the intermediate matrix files, and later blocks are still running.
The numerical job retains its original prefix-sampled contexts; primary fitting
uses the separately corrected shuffled bank. The primary launch gate still waits
for the complete registered all-layer result. Reproduce this snapshot with
`scripts/audit_numerical_checkpoint_cosines.py --run runs/numerical-proposal-v2
--max-layer 6 --output artifacts/numerical-partial-cosine-audit.json`.

## Analytic horizon limitation (not a Qwen measurement)

At the final block, antithetic interventions give exactly
`Delta y_gamma = w_gamma(0 | t) delta`: only the source position changes. The
normalized horizon weight depends on source position. Therefore one global
matrix can have low coverage even when every conditional map is perfectly linear.
For uniform valid source positions, zero-mean perturbations independent of
position, and the same nonzero covariance at every position, the population
optimal matrix is `E[w] I`, with `R2 = E[w]^2 / E[w^2]`.

| Horizon gamma | Optimal global population R² | Upstream identity population R² |
|---|---:|---:|
| 0 | 1.0000 | 1.0000 |
| 0.5 | 0.9903 | 0.0557 |
| 0.9 | 0.6101 | -29.6142 |
| 1.0 | 0.1542 | -61.3812 |

These are analytic values, not estimated effects; sampling CIs do not apply.
The upstream identity targets a sum, whereas these responses use normalized
horizons. Its poor scores illustrate estimand mismatch, not a discovery of
inferior causal understanding. Donor perturbations may depend on position and
need not obey the assumptions, so this table is not their performance ceiling.
The specification's estimator and horizons remain unchanged. Low global lens
coverage alone cannot establish curvature or absent workspace structure.

[Diagnostic artifact](artifacts/final-horizon-diagnostic.json),
[reproduction script](scripts/check_final_horizon.py). Tests check the exact
position-dependent map against a tiny Qwen suffix and the population formula
against a balanced positive/negative basis dataset.

## Fitting-corpus sampling correction before primary execution

A corpus audit found that the original first-eligible-row sampler assigned 15
of 16 source paragraphs to *Valkyria Chronicles III* and 14 of 16 donor paragraphs
to *Tower Building of the Little Rock Arsenal*. This was a narrow, mismatched
bank, not a representative random selection from WikiText train. The waiting
primary controller was stopped before any primary fits or intervention collection.
The old benchmark and numerical-validation contexts are retained as exploratory
provenance; the running numerical test is unchanged.

The replacement sampler uses a fixed CPU-generator seed of 101 to permute all
row indices, then takes eligible rows in that order (at least 600 characters and
128 tokens). It still truncates each row to 128 tokens. Source and donor banks
are disjoint consecutive portions of the same permutation; J/R/J++ receive their
32-row union. This is a documented sampling-method correction based on corpus
structure, with no evaluation labels or outcomes used. The new 16-row source
and donor banks each contain 16 distinct articles. This small budget still limits
generalization; article diversity does not establish representativeness.
See [the reproducible audit](artifacts/wikitext-sampling-audit.json) and
[sampler audit script](scripts/audit_wikitext_sampling.py).

New row provenance and resumable-experiment fingerprints include the sampling
policy. Old caches reject the changed protocol; primary execution uses a new
output directory and a fixed checkout. Tests verify deterministic selection,
disjoint banks, baseline union equality, eligibility, and unchanged global RNG.
The numerical job launched at commit e9e5c90 must be resumed from that commit if
necessary; new sampler code must not be used with its existing checkpoints.

## Specification corrections and a failed graph criterion

R-lens validation now checks backward rules directly in addition to preserving
forward outputs. The RMSNorm VJP is compared with the detached-denominator
formula and distinguished from the ordinary normalization gradient. The gated
MLP VJP is compared with the explicit half-weighted sum of the gate and up paths,
using the detached SiLU sigmoid factor, including a zero gate activation with
nonzero up activation. Tests also verify method restoration after an exception.
These checks pass without changing the implementation or any fitted artifacts;
they validate the specified heuristic rules, not their causal faithfulness.


The web/chat corpus iterator now filters document kind before tokenization. The
old pair of streams tokenized documents of both kinds before discarding the
unwanted kind, making the first chat window scan and tokenize the entire preceding
web portion. The corrected iterator preserves split membership, window tokens,
window hashes, ordering, deduplication and mixture weights. A regression test
compares every output window and provenance record while verifying that only the
selected document kind is tokenized. The waiting corpus builder will use this
validated loader for its budget audit; document-count and token-budget settings
are unchanged. No model data or evaluation outcomes motivated this efficiency
change, and no production throughput speedup has been measured yet.


Primary cross-validation now reuses zero-penalty candidate scores across priors.
At λ=0 the prior term vanishes for ridge, Huber IRLS and prompt-block MoM, so
repeating their fold fits for I/J/R adds computation without changing the
estimator. All candidate rows remain present, their order is unchanged, and ties
retain the original first-candidate rule. A reference test covers all three
estimators and verifies exact candidate-score and winner equality plus equal
final matrices; 30 redundant fold fits are removed in that test. Nonfinite CV
predictions now fail explicitly instead of entering candidate selection. No
production fitting outcomes motivated the optimization. Only the waiting primary
controller is requeued from the validated source; numerical validation continues
on its original checkpoint convention. No GPU wall-time savings are claimed yet.


Contrastive split-half nulls now reuse per-prompt activation/readout sums and
batch the vocabulary projections. The same seeded prompt permutations are
retained; normalization uses each half's actual position count, so unequal
numbers of positions per prompt preserve the original sample-weighted estimand.
Tests compare both readout modes with the unbatched reference and verify the
reduced readout call count. No GPU wall-time speedup is claimed yet.

Flat observed contrasts now retain their deterministic target rank but have
undefined rank significance. A constructed zero-contrast test shows why: the
target's ID tie-break yields observed rank 1 while every control split yields
rank 2, giving a rank-only p-value of 1/1001 despite zero effect. Such a value
would not justify concept detection. The new output records
`rank_test_status=undefined_flat_contrast` and a null p-value, preserving all
1,000 null ranks. Nonflat contrasts retain the specified rank-null comparison.
The planted runner's protocol fingerprint changes. No production contrastive
results preceded this correction.


Held-out faithfulness evaluation now shares each antithetic forward pair and its
per-position token log probabilities across all pending horizon kernels. The old
runner reset the same sampling seed and repeated those calculations once per
horizon. The source/donor draws, predictions, bootstrap groups and estimands are
unchanged. With all four horizons pending, this removes fourfold duplication of
branch evaluation and actual-log-probability computation, not all evaluation
work; no GPU wall-time speedup is claimed yet. Per-horizon matrices are retained
on the device for the current layer. Tests compare the shared reductions with
independent per-position definitions, count branch/readout calls, and verify
unchanged-cache resume and changed-matrix rejection. The cache protocol changes
because fp32 reduction ordering can differ. No primary evaluation result was used
to choose this implementation change.


Sparse-pursuit stopping previously compared the best new atom's squared projection
with the random-dictionary threshold before applying the constrained gradient
step. Those gains need not equal the actual marginal reconstruction improvement.
A deterministic synthetic counterexample (seed 18, dimension 16, 40 atoms,
iteration 6) gives candidate gain 1.02693 but achieved gain 0.85648. The corrected
procedure forms the candidate step, measures the actual reduction in squared
residual norm, and accepts only if it exceeds the existing 95th-percentile null.
The null remains the best nonnegative one-atom gain from an isotropic random
dictionary of matched size and row norms, sampled by its exact order-statistic
distribution. No evaluation labels select this rule. Tests force a 0.9 threshold
to verify rejection of the counterexample step and preservation of the prior
coefficients, and verify an empty decomposition for a zero dictionary. Readout
results identify the new pursuit protocol. No production sparse readout results
preceded this correction.
The first counterexample fixture failed validation because its diagnostic search
continued after a zero-gain stopping point. That unreachable fixture (seed 14)
was rejected and replaced with the reachable seed-18 example. Commit 6b9b159
was pushed before its failing test result was checked; the follow-up commit
records and corrects the fixture. The actual-gain acceptance rule was not relaxed.


A dedicated all-layer transcoder benchmark now precedes the 100M-token training
stage in the full-study runner. It performs three warmup and ten measured updates
with the registered width, TopK, Adam optimizers and feature-activity tracking,
then discards the benchmark weights. It reports raw update times, optimizer-state
bytes, peak CUDA memory and a linear update-time projection. WikiText is only a
timing proxy: the projection excludes web/chat tokenization, corpus auditing,
deduplication, checkpoint I/O and evaluation. These exclusions prevent treating
it as a total completion-time estimate. A tiny integration test verifies all-layer
optimizer allocation and the projection arithmetic. The production GPU benchmark
has not yet run; it must be measured before the full transcoder training launch.


Transcoder evaluation now computes centered float64 reporting moments from fp32
model outputs, replacing cancellation-prone subtraction of raw fp32 moments.
The model, gradients and training remain fp32. Window-level moments merge with
between-window mean corrections, including repeated windows in document bootstrap
samples. Zero output variance and equal clean/ablated cross-entropy produce
explicitly undefined ratios rather than NaNs that would fail JSON serialization.
Intervals require at least 20 finite bootstrap replicates; undefined replicate
counts are retained. At least two distinct held-out documents are required for
document-bootstrap intervals. Held-out moments, window provenance and all three
per-window CE measurements are saved for independent reaggregation. Tests cover
small variance on a large offset, repeated windows, constant outputs, equal CE
and undefined resamples. A tiny end-to-end training/evaluation test loads the
saved window statistics and independently reproduces each reported layer variance
and the all-MLP CE-recovery ratio. The evaluation protocol changes; no production Qwen
transcoder measurements preceded this correction.


Faithfulness-cell caches now bind the actual lens-matrix hashes, primary fitting
metadata, and source/donor row provenance. Previously the experiment fingerprint
covered evaluation settings but did not identify the matrices, so changing a
fitted artifact could silently reuse old cell measurements. Each cell now records
its input fingerprint and a provenance sidecar; missing or changed bindings raise
an error instead of reusing the result. An integration test resumes an unchanged
cell without recomputing effects, then changes its ILS matrix and verifies that
resume is refused. The protocol version changes; no production faithfulness
results preceded this correction. The all-layer audit also verified that the
J++ fitter already includes an explicitly labeled analytic final-block identity
under the upstream summed-target convention.


The measured-table exporter previously omitted nested dictionaries containing
fields beyond `mean` and `ci`. This would have dropped undefined-case counts,
selected prior/penalty/estimator details, native-estimand intervals and polar
diagnostics from generated summaries. It now preserves these as dot-separated
columns, retains short interval arrays and null values, and rejects ambiguous
column-name collisions. Long raw-sample arrays remain in their original result
files. Each exported stage includes the source result file's SHA-256 digest.
Regression tests check preservation of negative R², undefined outcomes, interval
bounds, fit selections and diagnostic quantities. This is a reporting correction;
no measured values or selection rules change.


The planted-organism model-difference diagnostic previously measured only held-out
WikiText contexts. It now additionally measures the matched behavioral prompts
and the four base-completion offsets used by the off-policy graph comparison.
Each perturbation is `h_tuned - h_base` at the same token context and site, and
its antithetic response runs through the base model with adapters disabled.
The fitted base ILS matrix predicts that response. Confidence intervals resample
prompts, keeping their four sites together. WikiText and behavioral-context R²
remain separate tables; neither is the fraction of the full tuned model's
behavioral trait explained. Short task prompts use skip0 and an appended,
excluded EOS sentinel, matching graph coverage and explicitly differing from
128-token skip16 fitting. A separate tiny-model test compares the response with
an independent copy of the base model and verifies adapter restoration. Input
length mismatches now raise rather than silently truncating paired contexts.
The run protocol changes, and the additional unbenchmarked time estimate is
14,400 seconds (28,800 total); actual stage timings remain required. No production
planted-organism results have been generated.


The Procrustes diagnostic now records raw expert-map dispersion, aligned-map
dispersion, positive-factor dispersion, and rotation chordal dispersion. For
square polar factors `A_k = U_k P_k`, the orthogonal solution makes
`U_k Q_k = U_0` by construction; aligned disagreement therefore reflects the
remaining `P_k` disagreement. This operation is useful to characterize the maps,
but its success does not identify rotation as the cause of expert disagreement.
Conversely, positive-factor disagreement can occur between purely linear maps
with different gains and is not itself evidence of curvature. A scaling-only
synthetic test preserves all disagreement after alignment; a mixed rotation and
scaling test verifies that aligned dispersion equals positive-factor dispersion.
The primary diagnostic still compares recall, interventional R² and effective
rank without selecting rotations using labels. Expert fitting provenance is
checked, and hashes of both expert and readout-state files bind cached results.
No Qwen Procrustes measurements have been completed.


Multilayer causal-gate auditing found that all-zero realized swaps were counted
as failed directional claims, and nonfinite scores could also silently become
failures. The band gate now retains zero-displacement interventions and
nonzero interventions with degenerate covariance as undefined (`passes=null`).
The observed score change remains recorded, but no random-direction comparison
is claimed. Nonfinite logits, transformed states, scores or covariances fail
explicitly. Pass-rate summaries report measured and undefined cases separately;
all-undefined groups have no pass rate. A new cache protocol prevents mixing old
and corrected results. Tiny-model tests instrument all 100 controls across two
layers and two positions, verifying the realized norm at every site and the
support of rank-one covariance draws. Control sizes match the observed lens
intervention's actual sequential displacements, including preceding layer
interventions; they are not recomputed from clean downstream states. No
production Qwen causal results preceded this correction.


Calibration auditing found that an unavailable GPD tail fit (fewer than 100
strict exceedances above the empirical 95th percentile) also incorrectly set
the empirical score standard deviation to zero. Discrete but varying reference
scores would therefore lose their z-scored readouts. The correction preserves
the actual standard deviation independently of tail availability. Such tokens
can have finite z scores, while their unsupported tail p-values remain
conservatively 1. Constant references retain undefined z scores. Nonfinite
reference scores fail explicitly. Production calibration artifacts are versioned,
and both the fitting and readout runners reject the old calibration version.
Tests cover rare exceedances, tied thresholds, constant scores and nonfinite
inputs. No production Qwen calibration results preceded this correction.


The token-change Spearman implementation previously applied normalization and
log-softmax to a horizon-averaged residual. This is not the actual averaged
change in token log probabilities. It now reads each perturbed final position
first, then averages its antithetic log-probability change with the horizon
weights. Lens predictions use the source state mapped by that lens, without
centering predictions on the observed final residual. The corrected protocol
has a new cache fingerprint. A regression test distinguishes these quantities.
No production Spearman results were generated with the old implementation.

Fitting-corpus exposure was also unequal: ILS used 16 source and 16 donor prompts
while fitted J/R/J++ used only the sources. All three fitted baselines now use
the same 32-prompt union. ILS retains the independent donor bank. Fold-specific
J/R priors use only training-fold sources plus the separate donor bank; held-out
source prompts are excluded. This equalizes corpus exposure, not the number or
distribution of derivative/intervention observations. J++ still additionally
uses labeled dev data and remains labeled accordingly. Configuration fingerprints
prevent reusing the old baseline shards as new-protocol artifacts.

Read-layer configurations now cover all 24 blocks. The transcoder configuration
now requests 100 million token presentations per layer. Its approximately
2.75-million-second estimate is a linear extrapolation, not a measured GPU cost.
Repeated presentations of the existing small corpus are not 100 million distinct
tokens. The trainer now preflights a finite, globally deduplicated training-window
stream and rejects overlap with validation windows before loading the model.
The existing corpus yields **2,557,568** distinct eligible tokens in the configured
mixture, so it correctly fails the 100M preflight. The larger pinned FineWeb and
UltraChat corpus has since been prepared and audited as described above;
no 100M training run has occurred.

`QwenGraph.replacement()` implements live transcoder features plus the fixed
clean reconstruction error, freezing only attention patterns and RMS
denominators. It reproduces clean outputs on tiny inputs. A finite intervention
can change downstream TopK support, however, so the local derivative need not
equal the finite effect. The reproducible untrained four-layer diagnostic in
[replacement-semantics.json](artifacts/replacement-semantics.json) records relative
error **0.26489**, failing the requested **<0.001** criterion. Downstream support
changes number 60, 60 and 62 coordinates. This is a synthetic counterexample to
equating a local derivative with an arbitrary propagated finite intervention,
not a counterexample to a direct edge with its mediated paths excluded. It is
not a trained-model causal or workspace claim. No random-direction claim is made.

A fixed scale sweep of exactly the same seeded fixture is preserved in
[replacement-scale-diagnostic.json](artifacts/replacement-scale-diagnostic.json).
It reproduces the original scale-one result exactly. Every nonzero tested scale
changes downstream TopK support and fails the unchanged 0.001 tolerance:

| Displacement scale | Relative derivative/effect error | Pass |
| --- | --- | --- |
| 0 | 0 | Trivial zero-intervention equality only |
| 0.000001 | 1.669865 | No |
| 0.00001 | 1.169741 | No |
| 0.0001 | 0.114572 | No |
| 0.001 | 0.203280 | No |
| 0.01 | 0.309693 | No |
| 0.05 | 0.273934 | No |
| 0.1 | 0.269201 | No |
| 0.25 | 0.266332 | No |
| 0.5 | 0.265372 | No |
| 1 | 0.264890 | No |

The smallest nonzero scale already changes 16 feature-support entries in the
final layer. Endpoint support counts alone do not prove where support changed
along the path, and this untrained fixture does not establish behavior in trained
transcoders. No production scale was selected from these results. Deterministic
fixture values have no sampling confidence intervals. The measured diagnostic
computation took 0.135 seconds (excluding interpreter/import startup) and its
artifact is 5,284 bytes, against pre-run estimates of 10 seconds and 20,000 bytes.

The previous independent-feature affine graph passes its own tests but uses an
extra freeze. Its existing results remain exploratory. Its runner now requires
an explicit `--exploratory-affine` flag. We have not relaxed the finite-edge
threshold, frozen additional gates in the new replacement, or represented the
failed criterion as success. Full graph attribution and budget closure under
the specified semantics remain unresolved. Tests explicitly preserve this
measured failure rather than requiring a false pass.

Inspection of [circuit-tracer's pinned implementation](https://github.com/decoderesearch/circuit-tracer/blob/255d989e93be1299d53d67b59e9fd76a0e274b14/circuit_tracer/replacement_model/replacement_model_transformerlens.py)
clarifies the estimand distinction: its direct attribution backward pass detaches
the non-skip MLP contribution, and constrained interventions freeze MLP outputs
while retaining skip changes. Thus our direct affine graph is structurally
consistent with that direct-effect convention. It is still distinct from a
propagated intervention in the live replacement with only attention/norms
frozen. Every checked exploratory edge now reports both measurements plus
original-model norms-only survival. The live result records support switches,
its own local-derivative error, and the direct-edge prediction error. A direct
edge passing does not assert that either propagated measurement passes.

The phrase dictionary now includes eligible targets from both the five readout
tasks and WorkspaceBench, while phrase evaluation remains on its declared
items. This fixes missing graph target rows caused by building the dictionary
only from WorkspaceBench. Penalties are still chosen solely by interventional
cross-validation. Targets outside 2–5 model tokens remain explicitly unsupported;
this change does not claim complete budget closure. The artifact audit now
explicitly rejects exploratory affine graph outputs as full proposal compliance
and checks distinct training tokens for the registered transcoder budget.

Readout exports now exclude masked/nonfinite entries rather than filling a
top-ten list with invalid token IDs when fewer than ten remain. Raw and z-score
dead-row rates are both recorded; an empty list has an undefined rate. The
transcoder trainer now persists training-wide feature-activation masks and
reports both never-active-during-training fractions and held-out inactivity.
Legacy checkpoints without this history cannot silently claim the former.

Checkpoint cadence changed from 128 to 8192 updates for the 100M-token run,
without changing data, optimization, or sample counts. The 24 transcoders contain
about 327.7M parameters; fp32 weights plus two Adam moments occupy about 3.93 GB
per training checkpoint. The former cadence implied about 6104 saves (24 TB of
writes); the new cadence implies 96 saves (378 GB), before serialization overhead.
These are engineering estimates, not observed training costs. The disk estimate
is now 10 GB to cover simultaneous old/new checkpoint files during atomic writes.

The prefix-invariance test now explicitly compares both intervention signs with
the clean output, on both the trained HMM and tiny random model. Equality between
the two signs alone could miss a shared error. The per-position suffix API also
rejects invalid source positions, including the excluded final token.

Readout ranks previously counted only strictly higher scores, which could
incorrectly assign many tied tokens rank one. Ranks and displayed top-k lists
now use descending score followed by ascending token ID (dictionary index for
phrases). An all-zero lens therefore cannot give every target rank one. Ranking
protocol versions prevent old cached evaluations from being presented as the
corrected protocol. No production Qwen readout outcomes selected this rule.

The phrase estimand has a structural null at the final block: a prompt-site
block-output intervention cannot affect token probabilities at later positions
after a fixed, nonempty probe suffix. A tiny-model regression test confirms zero
effects at several positions. When the observed fitting responses are all
exactly zero, every configured zero-prior ridge penalty gives the same zero
minimum-norm row; the artifact records that row with **no selected penalty and
undefined interventional R²**. This does not infer absence of sub-precision
effects at other layers. A zero direction or degenerate covariance has an
undefined causal gate, counted separately from measured failures. A zero graph
target has an undefined 0/0 attribution share and is never classified as dark
matter solely for that reason. Other nonzero constant responses still cannot
select a penalty by R² and are not silently assigned one.

The coordinate-swap implementation now uses the stated Moore–Penrose formula
for singular as well as full-rank direction matrices. The previous full-rank
restriction was unnecessary. Singular swaps need not be involutions; no
replacement formula or evaluation-based regularizer was introduced.

Phrase ridge fitting now shares each fold/penalty factorization across phrase
right-hand sides while selecting penalties independently by the same per-phrase
R². A small fp32 synthetic comparison retained all selected penalties and
matched scalar-reference matrices within 9.54e-7 absolute error. Three
alternating timed repeats took 0.0600–0.0610 seconds for scalar fits and
0.00308–0.00314 seconds for shared fits; these are raw engineering timings,
not confidence intervals or predictions of Qwen GPU speed. The reproducible
[benchmark artifact](artifacts/phrase-fit-benchmark.json) records inputs and
source hashes. Data budgets, folds, candidate penalties and estimators did not change.

## Model and numerical conventions

The subject is **Qwen2.5-0.5B-Instruct**, 24 blocks, residual width 896, vocabulary
151936, tied embeddings, Qwen2 RMSNorm and gated SiLU MLPs. The local checkpoint
is 999599221 bytes, stored in BF16 without quantization and loaded in fp32.
There is no q/k normalization or linear-attention block. GPU TF32 is disabled.
Machine-specific inventory stays outside this public repository.

Fitting sites are outputs leaving a block. Sequences have 128 tokens; sources
exclude the first 16 and final position. The target is the final block output
before final normalization. The actual reference `jlens.fit` **sums** future
targets before averaging source gradients; ILS uses a **normalized** horizon.
These are different estimands. The isotropic-limit check therefore compares to
our own horizon-matched Jacobian, never relabels the upstream sum as a mean.
Reference layer validation excludes the final block as a source; its summed
Jacobian is analytically identity. J++ uses the upstream API and is always
marked **label-tuned** when its combination weights have used dev labels.

A subsequently discovered [released exact-model J-lens](https://huggingface.co/MildHotSauce/jlens-qwen2.5-0.5b-instruct)
is imported only as **J-released-extra (200 prompts)**. Its artifact verifies
23 matrices of shape 896×896 and 200 fitting prompts. The public producer calls
the unmodified upstream fitter, whose inspected code verifies block-output
sources, final pre-norm targets, skip16, final-position exclusion and summed
future targets. The release stores FP16 matrices and its producer loads the model
in BF16; fp32 conversion does not recover that precision. The exact execution
command and model-weight revision are not embedded in the release. These limits
and its different corpus preparation/budget remain explicit in every comparison.
The extra baseline is ineligible as an ILS prior.

## Measured findings

The full numerical tables and bootstrap intervals are preserved in
[RESULTS.md](RESULTS.md) and [TOY_RESULTS.md](TOY_RESULTS.md), with the tensors and
machine-readable results under [artifacts](artifacts/README.md). Those documents
include earlier pilot limitations; this report supplies the current status.

- The four-block HMM model learned report behavior: held-out STATE-report
  accuracy 83.48%, CE 1.5292, after 204800 training token presentations.
- An exact Bayesian filter on 4000 independent HMM sequences achieved 92.716%
  [92.624%, 92.803%] state accuracy at 363711 valid non-report positions. Its
  posterior-expected accuracy was 92.722%. This is a separate observability
  diagnostic, not a lens result or a training target: uncertainty about the hidden
  state remains even with the correct generative model.
- In its normalized-horizon pilot, ILS interventional R² was approximately
  .08–.14; J/R confidence intervals overlap ILS. **No improvement is established.**
- Any-layer toy recall@1 was 0% for ILS, 30% [17.5%, 45.0%] for upstream J,
  and 72.5% [57.5%, 82.6%] for **J++ (label-tuned)**. This is not a controlled
  superiority comparison: horizon reduction, source-position sampling and layer
  bands differ. The 14-token vocabulary makes recall@10 permissive.
- With 100 matched random controls per toy site, ILS passed 2/24 tests,
  J-reference 2/24, R-reference 3/24, **J++ (label-tuned)** 0/24, logit 0/24.
  Eight prompts and overlapping intervals do not establish method differences.
- Four toy skip transcoders explained .707/.616/.792/.823 of held-out MLP-output
  variance. Replacing all MLPs recovered .710 [95% CI .641, .754] of the CE
  degradation relative to zeroing all MLPs. Distinct training tokens: 2048/layer.
- The trained nonlinear toy passed the normalized-J isotropic-limit thresholds
  at every block and all three registered epsilons. This does not substitute for
  the still-unrun full-matrix Qwen limit test.

![Toy interventional R²](artifacts/plots/toy-r2.png)

![Qwen finite-difference pilot](artifacts/plots/qwen-jvp.png)

### Budget closure and dark-matter cases

| Evaluation | Lens coverage | Graph completeness | Dark matter |
|---|---|---|---|
| Toy: 8 items × 4 sites, fixed position 40 | R² range −38.357 to −0.674 | Absolute source-attribution share .906–.974 | 0/32 at fixed .5/.5 thresholds |
| Qwen evaluation items | Not measured | Not measured | Unknown |

The toy's poor lens coverage at these task sites is a negative finding, even
though graph completeness is high. Its 0/32 dark-matter count is **not** zero
unexplained computation: the rule requires both quantities to be low. Its
bootstrap interval [0, 0] is a finite-sample degeneracy, not population certainty.
The explicit toy dark-matter case list is empty under those fixed thresholds;
all 32 individual cases are recorded in `artifacts/toy-budget/results.json`.
Unexplained lens variance is `1 − R²` and is not clipped when R² is negative.

The existing toy graph numbers freeze attention, norm denominators and independent
feature-source activations. They measure an affine source-to-target graph, not
causal faithfulness of the original nonlinear network. The new graph evaluator
additionally checks decoded source-output patches in the **original** model with
only norm denominators frozen; this newer workflow has tiny-model tests but no
production Qwen results yet.

## Answers to the eight research questions

| Question | Supported answer |
|---|---|
| 1. Does ILS beat J/R, especially early? | Unknown for Qwen. Toy R² intervals overlap; no improvement claim. |
| 2. Does unsupervised ILS match label-tuned J++ readout? | Not in the observed toy top-1 measurements, subject to the stated estimand/band confounds. Unknown for Qwen. |
| 3. Do lens interventions beat the random gate? | Occasionally for toy ILS/J/R; no broad selectivity result. J++ (label-tuned) passed none of its 24 toy tests. Qwen unknown. |
| 4. Does the concept lens separate horizons? | Qwen readouts not measured. Algebraically, the specified residual vanishes at eta=0 when the short-horizon map and covariance are full rank. Regularized residual content alone cannot establish a distinct concept space. |
| 5. Which tool detects the planted trait, and how much is outside the workspace? | No Qwen organism has been trained yet. Adapter training/resume and control restoration pass tiny tests. Model-diff R² will measure the base model's response to those perturbations, not total behavioral trait variance. |
| 6. Is early expert disagreement rotation or curvature? | Unknown. Fixed-expert Procrustes diagnostics and epsilon sweeps are implemented. Source-position-dependent horizon normalization is an additional confound. |
| 7. What fraction is dark matter? | 0/32 in the limited toy budget diagnostic; unknown for Qwen. |
| 8. Does this model have workspace-like structure compared with larger models? | No defensible conclusion yet. Published larger-model readout numbers cannot establish this model's causal workspace. |

### Published comparison points

Gurnee et al. report top-1 intermediate-swap success on 50 two-hop factual
prompts. These are published reference results, not measurements from this
repository. [Original paper, Figure 15 and accompanying text](https://transformer-circuits.pub/2026/workspace/index.html).

| Model | Published top-1 swap success | 95% interval |
|---|---|---|
| Claude Haiku 4.5 | 54% | Not supplied in the cited text |
| Claude Sonnet 4.5 | 70% | Not supplied in the cited text |
| Claude Opus 4.5 | 70% | Not supplied in the cited text |
| Qwen2.5-0.5B-Instruct, this study | Not measured | Not measured |

Our matched-random gate pass rate is a different metric from top-1 swap success;
the toy's 2/24 ILS gate passes cannot be compared numerically to these percentages.
Task filtering, layer bands, model families and control protocols also differ.
The paper identifies model-size scaling as an open question, so this source does
not establish a parameter threshold at which workspace structure must appear.
[Discussion of scale](https://transformer-circuits.pub/2026/workspace/index.html).

## Compute and execution status

| Phase | Estimate / actual evidence | Status |
|---|---|---|
| Inventory and Qwen smoke | fp32 forward/backward completed; about 3.65 GiB peak RSS | Complete |
| Qwen finite-difference pilot | 14.51 seconds CPU, one synthetic sequence | Complete, numerical pilot only |
| Upstream J/R benchmark | J 223.35 s; R 221.39 s per prompt on CPU | Complete benchmark |
| Main Qwen collection | Cached-suffix estimate ~2.94 CPU-days, plus baseline fitting | Stopped; 9 baseline prompt shards saved, no intervention collection |
| Toy upstream J/R/J++ | 13.11 seconds | Complete |
| Toy matched causal controls | 19.35 seconds | Complete |
| Toy transcoders/graphs/budget | 3.35 seconds | Complete |
| Toy isotropic-limit | 1.02 seconds | Complete |
| HMM Bayesian observability diagnostic | 10 seconds estimated / 0.87 seconds actual, CPU | Complete; no model fitting |
| New production stages | Config estimates are unbenchmarked GPU budgets, not actual costs | Not run |

Every new production runner writes its estimate before expensive work, records
its protocol and checkpoint provenance, and records elapsed time and artifact
size. GPU runners reject busy devices and never silently fall back to CPU.
The current 131 small-input tests pass. This includes numerical identities,
reference conventions, LRP forward preservation, LoRA base-weight preservation,
transcoder substitution, graph finite interventions, paired statistics and
checkpoint resume. These checks do not replace end-to-end Qwen measurements.

## Deviations, corrections and limits

1. **Target convention conflict:** upstream summed targets versus normalized ILS
   horizons are separately labeled. Equivalence is checked against our own J.
2. **Compute:** the planned main ILS budget is 8d=7168 interventions per layer and
   epsilon, drawn across 16 source prompts. Repeated interventions are not new
   prompts. The large CPU run was stopped by instruction; GPU runs await capacity.
3. **Toy pilot:** d=16, 128 fitting interventions, one model seed. J++ uses PCA16
   because PCA64 is impossible. Baseline source-position sampling differs from
   the original ILS pilot and is documented rather than concealed.
4. **CV leakage correction:** the initial pilot reused full-data J/R priors in
   validation folds. Corrected results recompute priors from training-fold prompts
   only, with unchanged seeds and search grid. No readout labels chose ILS.
5. **Isotropic-limit design:** per-context randomly rotated orthogonal bases
   provide balanced covariance. They are marginally isotropic, not iid draws.
6. **Graph:** independent clean feature-source activations are an additional
   affine-graph assumption. Bias remainder counts as unexplained attribution.
   New norms-only survival uses original attention/MLPs; old toy survival kept
   feature sources fixed. Neither is silently described as an unfrozen graph.
7. **Transcoder scale:** the previous production configuration requested 524288
   token presentations/layer. The corrected configuration requests 100M; neither
   production budget has been run. Web/chat windows are explicitly interleaved;
   distinct tokens, presentations and source mix are separately counted.
8. **Capability/judging:** exact-match gates are conservative; no LLM judge is
   configured. WSB's registered banks and renderers are used, including association
   v5. This project uses temperature .8, while WSB used .7. Strict exclusion of
   intermediates in output conflicts with poetry targets; no exception is hidden.
9. **Hallucinations:** WSB has moved this evaluation to HallucinationBench. Its
   published index belongs to another model and will not be copied as our score.
   Absence of a token from an output is not treated as a contradiction.
10. **Phrases:** targets must tokenize to 2–5 tokens. Longer role-binding phrases
    are explicitly unscoreable. The whitened baseline uses three fixed synthetic
    word-assembly contexts; each ILS phrase penalty is selected independently by
    prompt-fold interventional R². No natural-language activation explainer is trained.
11. **Subliminal attempt:** at most five predeclared concepts, 256 candidate
    teacher prompts/arm and LoRA students for three seeds. Both upstream number
    filters must agree. Matched prompts are evaluated with an exact paired test,
    requiring a positive effect and p<.01 in every seed. This is a small adaptation,
    not paper-scale replication; failures will not be tuned away.
12. **Sparse pursuit correction:** the initial helper was matching pursuit. It is
    now nonnegative gradient pursuit, with an active-support line search. Its null
    sampler uses the exact maximum-projection distribution for isotropic random
    dictionaries of the same size and norms, avoiding giant temporary allocations;
    an explicit-random-dictionary test checks this distribution.
13. **Token-change Spearman:** for gamma>0 the new runner reads the horizon-averaged
    final residual. That is a derived distribution, distinct from averaging actual
    next-token log-probability changes at individual future positions. Gamma0 uses
    the same-position final state. Both the estimand and limitation are recorded.
14. **Learning curves and ablations:** nested sample subsets preserve prompt
    diversity. Variants are compared on a common held-out donor/antithetic target,
    with their own native-target R² reported separately. This prevents changing the
    evaluation estimand from masquerading as an ablation improvement.
15. **Concept-lens identifiability:** for invertible `A_S` and positive-definite
    `Sigma`, setting eta=0 gives `B = A_L inverse(A_S)` and hence
    `A_L - B A_S = 0`. A numerical test verifies this identity. The specified
    regularized method is retained, and the evaluator records its remaining
    state-norm fraction; a nonzero residual need not represent uniquely
    long-horizon information.
16. **Gaussian qualification:** a Gaussian activation law supplies a sufficient
    setting for the Stein-identity link between mean derivatives and least
    squares. An assertion of equality *only* for Gaussian activations is too
    strong for a particular map: any linear map gives equality under arbitrary
    full-rank activation covariance. Donor perturbations also depend on source
    activations; their small-epsilon fit need not equal an unweighted Jacobian.
17. **Cross-benchmark leakage:** copied WorkspaceBench items are excluded from
    the common evaluation pool using both canonical IDs and normalized prompt
    hashes from J++'s labeled dev data. A namespace change does not make an item
    held out. Calibrations are bound to a hash of the exact fitted matrix.
18. **Task-site coverage:** the graph runner preserves the benchmark's exact
    input IDs and every prescribed read position. For its coverage diagnostic it
    appends a causally later EOS sentinel, excludes that sentinel from the target
    horizon, and uses skip0 on the original task context. This makes short and
    final-token sites measurable; it is explicitly a different evaluation
    distribution from the unchanged 128-token/skip16 WikiText fit. A test checks
    preserved states and the final-block identity at an original final-token site.
    Phrase targets without a fitted phrase row remain unknown, not complete.
19. **Role-binding scoring:** only complete capability-filtered pairs enter the
    exported pool. Phrase recall requires both directions and all scoreable
    targets in a pair; confidence intervals resample pairs. Other phrase families
    are reported separately with item-level resampling.
20. **Planted graphs:** matched base/tuned graphs use exactly the base model's
    sampled token IDs. Graphs decompose a specified readout, so their target
    contrast is not independent detection evidence beyond that readout. Changing
    model weights changes both source activations and transfer functions; the
    graph comparison does not claim that one feature causes the learned trait.
21. **Capability fallback:** a fixed 36-item LLM-authored auxiliary bank was
    prepared before any production Qwen capability/readout results. It is used
    only for a family with fewer than five retained reference test items, and
    must pass the identical capability and zero-leakage rules. Results carry a
    `:simplified` task suffix and are never silently pooled with upstream scores.
    Even simplified poetry is excluded when its successful answer reveals the
    target; the protocol conflict is retained rather than waived.
22. **Undefined measurements:** an explicit all-zero-effect test exposed NaN R²
    and failed strict JSON serialization. Constant-target R² now reports `null`
    with an explicit reason, not a perfect fit or zero. Undefined rank correlations
    and bootstrap replicates are counted. Cross-validation refuses constant
    targets because its registered R² selection objective is undefined there.
23. **Released-baseline search correction:** the earlier search of named lens
    repositories was incomplete. The broader exact-model search found the
    200-prompt artifact described above. It was downloaded and checked for finite
    tensors, layer indices, dimensions and checksum; its public producer and
    upstream conventions were inspected. No main fitting budget or method was
    changed, and the original reduced precision remains a comparison limitation.
24. **Graph expansion:** direct source-to-target attribution is now supplemented
    with bounded upstream feature expansion (at most eight feature targets,
    depth two). Active feature encoder rows use their clean TopK support under
    the affine freeze; norms-only survival recomputes the full TopK encoder.
    Every checked expanded edge must pass the same finite-intervention tolerance.
    Unexpanded features are listed; their ancestors are not counted a second
    time in root completeness. This is not an exhaustive graph enumeration.
25. **Phrase/read-site integration:** before any production phrase run, block 0
    was added to its registered layers so phrase rows cover the same four blocks
    used by WorkspaceBench and task graph evaluations. The unbenchmarked compute
    estimate increased accordingly; this choice uses no evaluation outcomes.

During this implementation pass, an older training/resume test failed because
its mocked behavior records omitted generated token IDs required by the new
graph export. The fixture now follows the actual generation schema. The complete
suite then passed; no research measurement or success threshold was changed.

The remaining production runs, calibration artifacts, stress results and full
coverage tables are required before this report can be called complete. An
implementation check, an unrun configuration and an unavailable measurement are
never reported as a negative experimental result.

Statistical tail fitting and bootstrap summaries use NumPy/SciPy native precision;
model evaluations, Jacobians, perturbations and regression tensors remain fp32.
The exact random-dictionary CDF uses float64 internally to avoid rounding its
near-one maximum quantiles to one at this vocabulary size.

The prepared transcoder corpus contains 4096 FineWeb documents and 512 UltraChat
conversations (4201 training / 407 validation documents; 16.34 MB). Revisions and
the content hash are in `artifacts/transcoder-corpus-manifest.json`; raw text is
kept out of the repository. The initial download duration was not instrumented
and is explicitly unknown, rather than reconstructed as a measured cost.
