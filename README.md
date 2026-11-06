# ILS-lens

Research code for fitting **Interventional Least-Squares lenses** to transformer
residual streams and measuring what their readouts explain.

**Status: experimental, incomplete.** The repository contains a working numerical
core, a four-layer toy HMM experiment, and a local Qwen2.5-0.5B numerical pilot.
It does not yet contain a completed Qwen benchmark, validated full attribution
graphs, or evidence that ILS outperforms existing lenses. Null results and
limitations are reported in [RESULTS.md](RESULTS.md).
The [toy research extension](TOY_RESULTS.md) includes upstream label-tuned J++,
random-direction controls, toy transcoders and per-item budget closure.

## Install

Use Python 3.11 or newer in a virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,plots]'
sh scripts/test.sh
```

[requirements-observed.txt](requirements-observed.txt) records the package
versions used for the included experiments. It is an environment record, not a
tested cross-platform lock. No GPU is required for the tests or toy experiment.

## Reproduce the experiments

```sh
sh scripts/toy.sh
python scripts/plot_results.py
```

The toy run trains a four-layer transformer on a seeded HMM, fits local J/R and
ILS maps on separate prompts, and evaluates held-out interventions and true-state
readout. Configs, fitted matrices, samples, and results are included. Regeneration
overwrites the corresponding local artifact files; copy them first if preserving
a run. Set `ILS_PYTHON` to use a different interpreter.

The Qwen pilot requires an existing, unquantized local model directory:

```sh
export ILS_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
mkdir -p logs
python scripts/smoke.py
sh scripts/qwen_pilot.sh
```

Weights are loaded locally and cast to fp32. Qwen weights and external datasets
are not redistributed here. To download reference repositories and WikiText-103
in a network-enabled environment, run `bash scripts/download_inputs.sh`.
Inspect the downloaded code before using it. Reference commit IDs are recorded
in `references/`; downloaded repositories remain untracked.

## Methods

For antithetic interventions δ at one block-output position, fit

```text
A = (mean(Δy δᵀ) + λ A₀) (mean(δ δᵀ) + λ I)⁻¹
```

The code supports normalized horizon weights, ridge/Huber/prompt-block MoM,
prompt-group cross-validation, and I/J/R priors without readout-label selection.
Additional experimental helpers cover calibration, concept residuals, contrasts,
random-direction controls, transcoders and phrase readout rows.

**Convention warning:** upstream `jlens.fit` averages source-position gradients
after summing over valid future targets. ILS uses a normalized future horizon
per source. These are different estimands. Their nonlinear equivalence has not
been established. See [RESULTS.md](RESULTS.md) and [REFERENCES.md](REFERENCES.md).

## Repository contents

- `ilslens/`: collection, fitting, readout and evaluation primitives.
- `tests/`: numerical, causality and tiny-model substitution checks.
- `configs/` and `scripts/`: reproducible experiment settings and commands.
- `artifacts/`: small toy weights, sample tensors, fitted lenses and measured results.
- `RESULTS.md`: research findings and limitations.

See [CONTRIBUTING.md](CONTRIBUTING.md) for research and contribution standards.
The original code is MIT licensed; third-party models, data and repositories
retain their own licenses. See [NOTICE](NOTICE).

## Primary Qwen run

The resumable J/R and donor-collection runner uses the preregistered settings in
`configs/qwen-main.yaml`: 16 source prompts, independent donor prompts, 7168
interventions per layer/epsilon, all 24 layers, six epsilons and four horizons.
On the measured CPU, collection is estimated at about three days. Inspect the
configuration and benchmark before launching on another machine.

```sh
export ILS_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
python scripts/run_qwen_main.py --dataset data/wikitext-103-raw-v1 \
  --reference references/jacobian-lens --phase benchmark
python scripts/run_qwen_main.py --dataset data/wikitext-103-raw-v1 \
  --reference references/jacobian-lens --phase all
python scripts/fit_qwen_offline.py --run runs/qwen-main
```

The runner saves atomic per-prompt/per-layer checkpoints under ignored `runs/`.
The offline fitter requires complete training and held-out collection and uses
training-fold-only J/R priors. These commands cover only the primary fitting and
intervention work, not the entire planned research study.

Production model runners now default to guarded CUDA execution. They reject a
busy GPU or insufficient free VRAM and never silently fall back to CPU. CPU
checkpoints remain reusable; newly written checkpoints record their execution
device. TF32 is disabled for fp32 research. These GPU paths require validation
on an idle device before accepting production GPU measurements.

Additional evaluation entry points include `evaluate_readouts.py` (strict
capability/leakage gates), `fit_jpp_qwen.py` (labeled dev only),
`calibrate_qwen.py` (100k reference activations), and `evaluate_contrast.py`
(on/off-policy contrasts and prompt-group null ranks). Run each with `--help`.
Modules for causal swaps, phrase continuation effects, LoRA adapters and stress
test statistics have small-input tests; their presence is not a completed
large-model experiment.

`run_planted.py` saves an indirect-example LoRA adapter, measures paired base/tuned
behavior, and can evaluate absolute, contrastive, and model-difference readouts.
`run_subliminal.py` uses the two upstream number filters and requires a positive
paired effect at p < .01 for each of three student seeds. The configuration
predeclares up to five concepts; a failed transmission remains a result.

`train_transcoders.py` consumes a versioned local web/chat corpus, trains all-layer
TopK skip transcoders, and reports held-out variance explained, dead features and
all-MLP substitution CE recovery. Its 524288-token presentation budget is below
the requested 100M target; distinct tokens and repeated presentations are separate.
`evaluate_graphs.py` validates feature, error and embedding edges, separately checks
interventions with only norms frozen in the original model, and emits itemwise
budget closure. `evaluate_causal.py` implements swap/ablation/answer controls and
100 covariance-matched random interventions at every patched site. These new
production experiments have not yet been run on Qwen.

The current [research report](REPORT.md) separates completed measurements from
unrun production stages and answers all eight research questions with the
available evidence. `evaluate_workspacebench.py` uses upstream registered banks
and renderers; `run_template_lens.py`, `evaluate_concept.py`,
`evaluate_faithfulness.py`, `check_qwen_limit.py`, `collect_ablations.py`,
`fit_diagnostics.py` and `evaluate_procrustes.py` implement their respective
registered measurements. They remain pending on production Qwen artifacts.
