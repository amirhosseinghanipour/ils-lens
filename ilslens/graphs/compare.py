"""Matched base/tuned graph diagnostics, without causal detection claims."""
import numpy as np

def compare_graphs(base,tuned,seed=0,repeats=1000):
    key=lambda row:(row['item_id'],row['layer'],row['position'],row['token'])
    def indexed(values):
        result={}
        for row in values:
            identity=key(row)
            if identity in result:raise ValueError(f'Duplicate graph site: {identity}')
            result[identity]=row
        return result
    first=indexed(base);second=indexed(tuned)
    if first.keys()!=second.keys():raise ValueError('Base and tuned graph sites differ')
    rows=[]
    for identity,a in first.items():
        b=second[identity]
        if not a.get('context_fingerprint') or a['context_fingerprint']!=b.get('context_fingerprint'):
            raise ValueError('Base and tuned token contexts differ or are unverified')
        measured=a.get('graph_status')=='measured' and b.get('graph_status')=='measured'
        if measured and (not a.get('target_fingerprint') or a['target_fingerprint']!=b.get('target_fingerprint')):
            raise ValueError('Base and tuned target vectors differ or are unverified')
        row=dict(item_id=identity[0],layer=identity[1],position=identity[2],token=identity[3],
            context_fingerprint=a['context_fingerprint'],base_graph_status=a.get('graph_status'),tuned_graph_status=b.get('graph_status'))
        if not measured:
            rows.append(dict(row,status='unmeasured_pair',difference=None,
                base_completeness=a.get('graph_completeness'),tuned_completeness=b.get('graph_completeness')))
            continue
        if not a['all_edges_pass'] or not b['all_edges_pass']:raise ValueError('Cannot summarize unvalidated graph edges')
        if not all(np.isfinite(r['target_value']) for r in (a,b)):raise ValueError('Nonfinite measured graph target')
        rows.append(dict(row,status='measured',target_fingerprint=a['target_fingerprint'],
            base_target=a['target_value'],tuned_target=b['target_value'],difference=b['target_value']-a['target_value'],
            base_completeness=a['graph_completeness'],tuned_completeness=b['graph_completeness'],
            base_far_upstream=a['far_upstream_share'],tuned_far_upstream=b['far_upstream_share']))
    measured_rows=[row for row in rows if row['status']=='measured']
    summaries=[];rng=np.random.default_rng(seed)
    for layer in sorted({r['layer'] for r in measured_rows}):
        selected=[r for r in measured_rows if r['layer']==layer];ids=sorted({r['item_id'] for r in selected})
        baseline=np.array([np.mean([r['base_target'] for r in selected if r['item_id']==identity]) for identity in ids])
        differences=np.array([np.mean([r['difference'] for r in selected if r['item_id']==identity]) for identity in ids])
        if len(ids)<4:
            summaries.append(dict(layer=layer,prompts=len(ids),status='insufficient_prompts_for_null'));continue
        draws=differences[rng.integers(len(ids),size=(repeats,len(ids)))].mean(1)
        null=[]
        for _ in range(repeats):
            order=rng.permutation(len(ids));half=len(ids)//2
            null.append(baseline[order[:half]].mean()-baseline[order[half:]].mean())
        summaries.append(dict(layer=layer,prompts=len(ids),difference=float(differences.mean()),ci=np.quantile(draws,[.025,.975]).tolist(),
            control_split_half_null=null,upper_tail_p=(1+sum(value>=differences.mean() for value in null))/(repeats+1),
            bootstrap_unit='prompt'))
    return dict(rows=rows,summary=summaries,measured_pairs=len(measured_rows),unmeasured_pairs=len(rows)-len(measured_rows),
        comparison_protocol='exact-inputs-retain-unmeasured-v2',
        interpretation='A graph decomposes a specified target; target contrast is not independent evidence beyond that readout.',
        limitations=['Different model weights change both the local transfer functions and source activations',
                    'Base-trained transcoders are reused; tuned reconstruction error is recomputed',
                    'No claim that changing the highest-attribution feature causes the learned preference'])
