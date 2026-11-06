"""Publish completed numerical measurements with source provenance and limits."""
import argparse,hashlib,json
from pathlib import Path
from ilslens.validation_gate import validate_numerical_result
from ilslens.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--cosine-audit',required=True)
    p.add_argument('--output',default='artifacts/qwen-numerical-validation.json');a=p.parse_args()
    root=Path(a.run);result=json.loads((root/'results.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    launch=json.loads((root/'launch.json').read_text());gate=validate_numerical_result(result,manifest)
    audit=json.loads(Path(a.cosine_audit).read_text())
    if audit['source_fingerprint']!=manifest['fingerprint']:raise ValueError('Cosine audit belongs to another experiment')
    keys=lambda rows:[(r['layer'],r['epsilon']) for r in rows]
    if len(set(keys(audit['rows'])))!=len(audit['rows']) or set(keys(audit['rows']))!=set(keys(result['rows'])):
        raise ValueError('Cosine audit does not cover exactly the numerical result cells')
    summary=[]
    for epsilon in manifest['protocol']['config']['epsilons']:
        rows=[r for r in result['rows'] if r['epsilon']==epsilon]
        stable=[r['matrix_cosine_float64_reduction'] for r in audit['rows'] if r['epsilon']==epsilon]
        summary.append(dict(epsilon=epsilon,passed_cells=sum(r['passes'] for r in rows),total_cells=len(rows),
            minimum_matrix_cosine_fp32=min(r['matrix_cosine'] for r in rows),minimum_matrix_cosine_reporting_float64=min(stable),
            minimum_heldout_top10_overlap=min(r['heldout_top10_overlap'] for r in rows),
            maximum_directional_jvp_error=max(e for r in rows for e in r['jvp_errors'])))
    write_json(a.output,dict(result=result,manifest=manifest,validation_gate=gate,summary=summary,source_commit=launch['commit'],
        source_results_sha256=hashlib.sha256((root/'results.json').read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        cosine_audit_sha256=hashlib.sha256(Path(a.cosine_audit).read_bytes()).hexdigest(),
        original_estimate=dict(seconds=manifest['protocol']['config']['estimated_seconds'],disk_mb=manifest['protocol']['config']['estimated_disk_mb']),
        final_directory_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        interpretation='Passes the registered normalized-horizon own-J equivalence test. Does not show superiority to J/R/J++, causal faithfulness, or negligible directional fp32 error.',
        sampling='Original prefix-sampled numerical contexts, distinct from the corrected shuffled primary corpus',
        uncertainty='Numerical validation thresholds on fixed contexts; extrema have no population CI. Per-prompt overlap values were not persisted by this run, so no overlap bootstrap interval is reconstructed.'))

if __name__=='__main__':main()
