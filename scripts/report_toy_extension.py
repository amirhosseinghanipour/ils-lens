"""Generate the source-backed extension report without rerunning experiments."""
import json
from pathlib import Path

reference=json.loads(Path('artifacts/toy-reference/results.json').read_text())
gate=json.loads(Path('artifacts/toy-gate/results.json').read_text())
budget=json.loads(Path('artifacts/toy-budget/results.json').read_text())
def interval(value,ci):return f'{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]'
text='''# Toy research extension

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
'''
names=list(dict.fromkeys(row['lens'] for row in reference['rows']))
for name in names:
    rows=[r for r in reference['rows'] if r['lens']==name and r['layer']=='any_available']
    by_k={r['k']:r for r in rows}
    text+=f"| {name} | {'0–2' if name.startswith('J++') else '0–3'} | {interval(by_k[1]['recall'],by_k[1]['ci'])} | {interval(by_k[10]['recall'],by_k[10]['ci'])} |\n"
text+='''
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
'''
for r in gate['summary']:
    text+=f"| {r['lens']} | {r['layer']} | {interval(r['pass_rate'],r['ci'])} | {r['n']} |\n"
text+='''
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
'''
for r in budget['transcoders']:
    text+=f"| {r['layer']} | {interval(r['variance_explained'],r['variance_explained_ci'])} | {r['dead_fraction']:.3f} |\n"
ce=budget['ce']
text+=f"\nClean CE: {ce['clean']:.4f}; all-transcoder CE: {ce['replaced']:.4f}; all-MLP-ablated CE: {ce['ablated']:.4f}. CE recovery: {interval(ce['recovered'],ce['recovered_ci'])}.\n"
text+='''
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
'''
for r in budget['rows']:
    ratio='undefined' if r['survival_ratio'] is None else f"{r['survival_ratio']:.3f}"
    text+=f"| {r['item']} | {r['layer']} | {r['lens_r2']:.3f} | {r['unexplained']:.3f} | {r['graph_completeness']:.3f} | {r['dark_matter']} | {r['edge_relative_error']:.2g} | {ratio} |\n"
text+=f"\nAll {len(budget['rows'])} checked edges pass: {budget['all_checked_edges_pass']}. Dark-matter fraction: {interval(budget['dark_matter_fraction'],budget['dark_matter_ci'])}.\n"
text+='''
Dark-matter case list: none under these declared thresholds in these 32 toy
item/layer cases. That is not a conclusion about Qwen or an estimate of all
model computation. The bootstrap interval is degenerate because no sampled
case qualified. Norm-only survival still holds feature sources fixed; it does
not represent a completely unfrozen intervention. Transcoder reconstruction
quality and graph completeness are different quantities.

## Compute and provenance

'''
text+=f"| Experiment | Pre-run estimate | Actual seconds |\n|---|---|---|\n| Reference toy lenses | 180 s | {reference['seconds']:.2f} |\n| Toy random gate | 180 s | {gate['seconds']:.2f} |\n| Toy budget closure | 180 s | {budget['seconds']:.2f} |\n"
text+='''
All experiments use fp32 CPU computation with two threads. Configs, seeds,
matrices, transcoders and per-item outcomes are under the corresponding
`artifacts/toy-*` directories. The graph run was repeated with unchanged data
and settings to add prompt-bootstrap CE/variance intervals; no evaluation-based
model or hyperparameter changes were made. An upstream attribute-name mismatch
was corrected before the reference readout evaluation completed.
'''
Path('TOY_RESULTS.md').write_text(text)
print('Wrote TOY_RESULTS.md')
