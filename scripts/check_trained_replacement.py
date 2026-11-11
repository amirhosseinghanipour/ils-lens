"""Numerical finite-intervention diagnostic with trained Qwen transcoders.

This checks local derivatives against full feature ablations under only the
specified attention/norm freeze. It does not redefine direct graph edges, select
an intervention scale, or claim semantic causality. Failed checks are retained.
"""
import argparse
import hashlib
import time
from pathlib import Path
import torch
import yaml
from ilslens.experiment import Experiment
from ilslens.graphs.qwen import QwenGraph
from ilslens.graphs.intervention import replacement_effect
from ilslens.model import load_local
from ilslens.transcoders.io import load_transcoders
from ilslens.eval.prompting import render_read_prompt


def check_sources(graph, layers):
    """Ablate the largest active feature at the last input position per layer."""
    position=graph.ids.shape[1]-1
    token=int(graph.clean_logits[0,position].argmax())
    target=dict(kind='logit',position=position,token=token)
    with graph.replacement(),torch.no_grad():
        reconstructed=graph.forward()
    max_error=float((reconstructed-graph.clean_logits).abs().max())
    torch.testing.assert_close(reconstructed,graph.clean_logits,rtol=2e-5,atol=2e-5)
    rows=[]
    for layer in layers:
        values=graph.clean_features[layer][0,position]
        feature=int(values.argmax());activation=float(values[feature])
        if activation<=0:
            rows.append(dict(layer=layer,status='no_active_feature',passes=None))
            continue
        displacement=torch.zeros_like(graph.outputs[layer])
        displacement[0,position]=-values[feature]*graph.transcoders[layer].decoder.weight[:,feature]
        result=replacement_effect(graph,target,layer,displacement)
        rows.append(dict(layer=layer,feature=feature,activation=activation,
                         status='measured',displacement_norm=float(displacement.norm()),**result))
    return dict(target=target,clean_reconstruction_max_abs_error=max_error,rows=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/trained-replacement-check.yaml')
    parser.add_argument('--transcoders',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--after-process',type=int,help='Wait for an existing GPU job before loading models')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    if args.after_process is not None:
        from scripts.run_primary_after_validation import process_alive
        while process_alive(args.after_process):time.sleep(15)
    # Bind this diagnostic to actual trained weights, not merely directory names.
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(args.transcoders).glob('layer-*.pt'))}
    run=Experiment(args.output,dict(config=config,transcoder_sha256=hashes,
        protocol='live-topk-full-feature-ablation-diagnostic-v1'))
    torch.manual_seed(config['seed'])
    model,tokenizer=load_local('cuda',min_free_gib=9.5)
    transcoders=load_transcoders(args.transcoders,model)
    if len(hashes)!=len(transcoders):raise ValueError('Expected one hashed artifact per layer')
    records=[]
    for index,prompt in enumerate(config['prompts']):
        rendered=render_read_prompt(tokenizer,prompt,prompt_format='chat')
        ids=tokenizer(rendered,add_special_tokens=False,return_tensors='pt').input_ids.to(model.device)
        graph=QwenGraph(model,ids,transcoders)
        record=check_sources(graph,range(len(transcoders)))
        record.update(item=index,prompt=prompt,token_ids=ids[0].tolist())
        records.append(record)
        run.event('prompt-checked',item=index,checks=len(record['rows']),
                  failed=sum(r['passes'] is False for r in record['rows']))
        del graph
    run.result(dict(items=records,interpretation='Deterministic implementation diagnostic; no population confidence intervals. Local-derivative finite-effect equality differs from direct-edge equality. No semantic causal claim or random-direction gate pass is implied. Failures do not authorize freezing additional features or changing the 1e-3 tolerance.'))


if __name__=='__main__':main()
