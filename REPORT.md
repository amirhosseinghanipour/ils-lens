# ILS-lens research report — incomplete Qwen study

Updated 2026-11-06. **The requested research criteria are not yet met.** Completed
measurements are the Qwen numerical/reference benchmarks and the trained toy HMM
experiments below. Production Qwen experiment runners are implemented but their
presence is not a result. Long CPU runs were stopped at the owner's request;
the available RTX 3060 remains occupied by another workload. Missing measurements
are unknown, not null findings.

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
The current 56 small-input tests pass. This includes numerical identities,
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
7. **Transcoder scale:** the production configuration uses 524288 token
   presentations/layer, below 100M. Web/chat windows are explicitly interleaved;
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
