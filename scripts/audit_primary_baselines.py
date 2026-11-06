"""Validate completed J/R artifacts without loading a model or refitting maps."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from ilslens.budget import baseline_count
from ilslens.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--launch',required=True)
    p.add_argument('--output',default='artifacts/qwen-primary-baselines.json');a=p.parse_args();start=time.perf_counter();torch.set_num_threads(2)
    root=Path(a.run);manifest=json.loads((root/'manifest.json').read_text());config=manifest['config']
    launch=json.loads(Path(a.launch).read_text());events=[json.loads(line) for line in (root/'progress.jsonl').read_text().splitlines()]
    completed=[r for r in events if r['stage']=='baselines-complete']
    if not completed:raise ValueError('Baseline stage has not reported completion')
    rows=manifest['source_rows']+(manifest['donor_rows'] if config['baseline_corpus']=='source_and_donor' else [])
    if len(rows)!=baseline_count(config):raise ValueError('Unexpected fitting corpus size')
    files=[]
    for name in ('J','R'):
        for prompt in range(len(rows)):
            path=root/'baselines'/f'{name}-prompt-{prompt:03d}.pt'
            saved=torch.load(path,weights_only=True,map_location='cpu');meta=saved['metadata'];maps=saved['matrices']
            if meta['fingerprint']!=manifest['fingerprint'] or meta['prompt']!=prompt or meta['source_row']!=rows[prompt]:
                raise ValueError('Baseline data provenance mismatch')
            if meta['label_tuned'] or meta['target_layer']!=23 or meta['source_convention']!='block_output' or meta['target_reduction']!='sum' or meta['execution_device']!='cuda':
                raise ValueError('Unexpected baseline conventions')
            if set(maps)!=set(range(24)):raise ValueError('Missing or extra baseline layers')
            if any(m.dtype!=torch.float32 or m.shape!=(896,896) or not torch.isfinite(m).all() for m in maps.values()):
                raise ValueError('Invalid baseline matrix shape, dtype, or values')
            if not torch.equal(maps[23],torch.eye(896)):raise ValueError('Analytic final-block identity differs')
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            files.append(dict(name=path.name,sha256=digest,bytes=path.stat().st_size,lens=name,prompt=prompt))
    timings=[r for r in events if r['stage']=='baseline-prompt']
    write_json(a.output,dict(status='complete_artifact_validation',primary_fingerprint=manifest['fingerprint'],source_commit=launch['commit'],
        configuration=config,fitting_rows=rows,reference_revision=manifest['reference_revision'],files=files,
        baseline_prompt_count=len(rows),matrices_per_lens=len(rows)*24,fp32_finite_all_layers=True,final_block_identity_exact=True,
        native_target_reduction='sum',source='block_output_zero_based',target='final_block_output_pre_norm',target_layer=23,
        label_tuned=False,execution_device='cuda',tf32=False,stage_elapsed_seconds=completed[-1]['elapsed_seconds'],
        per_prompt_fit_seconds={name:[r['seconds'] for r in timings if r['lens']==name] for name in ('J','R')},
        original_estimated_seconds=config['baseline_estimated_seconds'],baseline_file_bytes=sum(r['bytes'] for r in files),
        audit_seconds=time.perf_counter()-start,
        limitation='All primary J/R fitting prompts completed; this is artifact/convention validation, not a held-out R2, readout, or causal result. Final-layer maps use the documented analytic identity. Other study phases remain incomplete.'))
    print(json.dumps(dict(files=len(files),bytes=sum(r['bytes'] for r in files),audit_seconds=time.perf_counter()-start)))

if __name__=='__main__':main()
