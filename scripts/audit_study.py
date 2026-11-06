"""Fail closed on missing research measurements; code files are not evidence."""
import argparse,json
from pathlib import Path
import torch
from ilslens.experiment import write_json
from ilslens.budget import baseline_count

def audit(root):
    root=Path(root);checks=[]
    def check(name,condition,detail):checks.append(dict(criterion=name,passes=bool(condition),detail=detail))
    primary=root/'qwen-main';manifest_path=primary/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    check('primary_manifest',manifest is not None,'Registered fitting data/conventions must be recorded')
    if manifest:
        config=manifest['config'];fingerprint=manifest['fingerprint']
        check('matched_fitting_corpus',config.get('baseline_corpus')=='source_and_donor',
              'J/R/J++ must see the same source-plus-donor training corpus as ILS')
        baseline_files=[primary/'baselines'/f'{name}-prompt-{i:03d}.pt' for name in ('J','R') for i in range(baseline_count(config))]
        check('all_J_R_baseline_prompts',all(p.exists() for p in baseline_files),dict(expected=len(baseline_files),present=sum(p.exists() for p in baseline_files)))
        for split,count in [('train',config['source_prompts']),('validation',config['heldout_prompts'])]:
            files=[primary/'samples'/split/f'prompt-{i:03d}-layer-{layer:02d}.pt' for i in range(count) for layer in range(24)]
            check('all_'+split+'_interventions',all(p.exists() for p in files),dict(expected=len(files),present=sum(p.exists() for p in files)))
        files=[primary/'fits'/f'layer-{l:02d}-epsilon-{e}-gamma-{g}.pt' for l in range(24) for e in config['epsilons'] for g in config['gammas']]
        check('all_selected_ILS_maps',all(p.exists() for p in files),dict(expected=len(files),present=sum(p.exists() for p in files)))
        invalid=[]
        for path in files:
            if not path.exists():continue
            artifact=torch.load(path,weights_only=True,map_location='cpu');meta=artifact['metadata']
            if artifact['matrix'].dtype!=torch.float32 or meta['fingerprint']!=fingerprint or meta['label_tuned'] or meta['N']<8*896:
                invalid.append(path.name)
        check('ILS_fp32_label_free_budget_provenance',all(p.exists() for p in files) and not invalid,invalid)
    jpp=root/'jpp/J++-label-tuned.pt'
    if jpp.exists():
        a=torch.load(jpp,weights_only=True,map_location='cpu');valid=a['metadata'].get('label_tuned') and len(a['matrices'])==24 and bool(a['metadata'].get('dev_item_ids'))
    else:valid=False
    check('J++_labeled_dev_baseline',valid,'All layers, explicit label-tuned metadata, nonempty dev IDs')
    check('released_exact_model_extra_baseline',(root/'released-baselines/released-jlens.pt').exists(),
          'Pinned released J-lens imported separately; unequal corpus budget and reduced original precision remain labeled')
    missing=[];invalid=[]
    for name in ('logit','J-reference','R-reference','ILS','J++ (label-tuned)','J-released-extra (200 prompts)'):
        for layer in range(24):
            path=root/'calibration'/name/f'layer-{layer:02d}.pt'
            if not path.exists():missing.append(f'{name}:{layer}');continue
            a=torch.load(path,weights_only=True,map_location='cpu')
            if a.get('reference_count',0)<100000:invalid.append(f'{name}:{layer}')
    check('all_100k_calibrations',not missing and not invalid,dict(missing=missing,invalid=invalid))
    required=['readouts','workspacebench','template','transcoders','graphs','causal','planted','planted-graphs','subliminal','numerical','faithfulness','diagnostics','procrustes','concept','contrast','hallucination']
    for name in required:
        path=root/name/'results.json'
        result=json.loads(path.read_text()) if path.exists() else {}
        check(name+'_measured_results',bool(result) and result.get('status') not in ('gates_only','collect_only','fit_only'),
              'Present' if result else 'No result artifact; an implementation is not a measurement')
    # Conditional requirements inspect substantive outcomes, not a success marker.
    for name,field in [('numerical','smallest_epsilon_passes_all_layers'),('readouts','complete_calibration')]:
        path=root/name/'results.json';a=json.loads(path.read_text()) if path.exists() else {}
        check(name+'_validation',a.get(field) is True,f'Requires {field}=true; failures must be investigated and reported')
    graph_path=root/'graphs/results.json';graph=json.loads(graph_path.read_text()) if graph_path.exists() else {}
    graph_rows=graph.get('budget',{}).get('rows',[])
    measured=[r for r in graph_rows if r.get('graph_status','measured')=='measured']
    check('graph_finite_edge_validation',bool(measured) and all(r.get('all_edges_pass') for r in measured),'Every measured graph row must have passing finite-edge checks; unsupported targets remain unknown')
    check('graph_all_targets_supported',bool(graph_rows) and len(measured)==len(graph_rows),'Unsupported phrase targets must not silently pass full budget closure')
    check('graph_proposal_semantics',bool(graph) and graph.get('exploratory_affine') is False,
          'Exploratory direct-affine results cannot satisfy the unresolved proposal freeze criterion')
    tc_path=root/'transcoders/results.json';tc=json.loads(tc_path.read_text()) if tc_path.exists() else {}
    training=tc.get('training',{})
    check('transcoder_registered_data_budget',training.get('token_presentations',0)>=100000000
          and training.get('distinct_tokens',0)>=100000000 and training.get('unique_training_windows') is True,
          'Registered 100M distinct-token configuration; no substitution of repeated presentations')
    planted_path=root/'planted/results.json';planted=json.loads(planted_path.read_text()) if planted_path.exists() else {}
    check('planted_detection_measured',bool(planted.get('model_diff_faithfulness')) and bool(planted.get('rows')),
        'A failed behavioral planting is a finding; detection measurements must still be reported conditionally')
    return dict(complete=all(c['passes'] for c in checks),passed=sum(c['passes'] for c in checks),total=len(checks),checks=checks,
        scope='Artifact completeness checks only; passing does not establish scientific validity or method superiority')

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs');p.add_argument('--output',default='runs/study-audit.json');args=p.parse_args()
    result=audit(args.runs);write_json(args.output,result)
    print(json.dumps(dict(complete=result['complete'],passed=result['passed'],total=result['total'],missing=[r['criterion'] for r in result['checks'] if not r['passes']]),indent=2))
    if not result['complete']:raise SystemExit(1)

if __name__=='__main__':main()
