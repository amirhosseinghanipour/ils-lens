"""Measured tables and figures, preserving missing values and negative results."""
import json,math,os,hashlib
from pathlib import Path

STAGES=('faithfulness','readouts','workspacebench','template','transcoders','graphs','causal',
        'concept','contrast','planted','planted-graphs','subliminal','diagnostics','procrustes','numerical','hallucination')

def cell(value):
    if value is None:return 'unknown / undefined'
    if isinstance(value,float):return f'{value:.6g}' if math.isfinite(value) else 'undefined'
    if isinstance(value,(list,dict)):value=json.dumps(value,ensure_ascii=False,allow_nan=False)
    return str(value).replace('|','\\|').replace('\n',' ')

def table(rows,columns):
    if not rows:return 'No measured rows.\n'
    return '| '+' | '.join(columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'+''.join(
        '| '+' | '.join(cell(row.get(key)) for key in columns)+' |\n' for row in rows)

def scalar_rows(rows):
    """Flatten nested statistics; retain intervals, omit long raw-sample arrays.

    Undefined counts, fit choices and diagnostic dictionaries are measurements,
    not disposable metadata. Dot-separated names preserve their parent context.
    """
    result=[]
    for row in rows:
        flattened={}
        def visit(key,value):
            if isinstance(value,dict):
                for child,item in value.items():visit(f'{key}.{child}',item)
            elif not isinstance(value,list) or (len(value)<=6 and all(not isinstance(item,(dict,list)) for item in value)):
                if key in flattened:raise ValueError(f'Ambiguous flattened measurement key: {key}')
                flattened[key]=value
        for key,value in row.items():visit(key,value)
        result.append(flattened)
    return result

def measured_tables(stage,result):
    if stage=='transcoders':return {'per_layer':result['transcoders'],'ce':[result['ce']]}
    if stage=='graphs':
        budget=result['budget']
        return {'budget':scalar_rows(budget['rows']),'dark_cases':scalar_rows(budget['dark_cases']),
            'aggregate':[{key:budget[key] for key in ('fraction','ci','known_cases','unknown_cases')}]}
    tables={}
    for key in ('rows','summary','causal_summary','interventional','readouts','interventional_faithfulness','model_diff_faithfulness','behavior_model_diff_faithfulness'):
        if key in result:tables[key]=scalar_rows(result[key])
    if stage=='subliminal':
        tables['per_seed']=[dict(concept=row['concept'],**seed) for row in result.get('rows',[]) for seed in row.get('seeds',[])]
    if stage=='planted':
        tables['absolute_and_model_diff_ranks']=[dict(lens=row['lens'],layer=row['layer'],mode=row['mode'],read_mode=row['read_mode'],
            base_rank=row['absolute_ranks'][row['read_mode']]['base'],tuned_rank=row['absolute_ranks'][row['read_mode']]['tuned'],
            model_diff_rank=row['model_diff_lens_rank'][row['read_mode']]) for row in result.get('rows',[])]
    if 'behavior' in result:tables['behavior']=[result['behavior']]
    return tables

def render(runs,output):
    runs=Path(runs);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    text=['# Qwen measured tables\n','These are generated measurements, not a declaration that the study is complete. '
        'Missing stages remain unknown. All J++ entries are label-tuned; the released 200-prompt J-lens is an extra baseline with a different budget and original precision. '
        'No method superiority is inferred from overlapping confidence intervals.\n']
    found={};missing=[]
    for stage in STAGES:
        path=runs/stage/'results.json';text.append(f'## {stage}\n')
        if not path.exists():missing.append(stage);text.append('Not measured.\n');continue
        result=json.loads(path.read_text());found[stage]=result
        tables=measured_tables(stage,result)
        summary=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fingerprint=result.get('fingerprint',result.get('metadata',{}).get('fingerprint')),tables=tables,
            seconds_this_invocation=result.get('seconds_this_invocation',result.get('seconds')),
            deviations=result.get('deviations',result.get('deviation')),interpretation=result.get('interpretation'))
        (output/f'{stage}.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        for name,rows in tables.items():
            text.append(f'### {name}\n');columns=list(dict.fromkeys(key for row in rows for key in row))
            text.append(table(rows,columns)+'\n')
        for key in ('deviations','deviation','interpretation','limitation','limitations','distinction','token_change_estimand','causal_claim_rule'):
            if key in result:text.append(f'{key}: {cell(result[key])}\n')
    text.append('## Execution evidence\n')
    costs=[]
    for path in sorted((runs/'study').glob('*.completed.json')):
        record=json.loads(path.read_text());costs.append(dict(stage=path.name.removesuffix('.completed.json'),seconds=record['seconds'],source_digest=record['source_digest']))
    text.append(table(costs,['stage','seconds','source_digest']))
    return '\n'.join(text),found,missing

def figures(found,output):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/ils-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);paths=[]
    def finish(fig,name):
        fig.tight_layout();fig.savefig(output/name,dpi=160);plt.close(fig);paths.append(name)
    if 'faithfulness' in found:
        rows=[r for r in found['faithfulness']['rows'] if r['epsilon']==.05 and r['gamma']==.9 and r['r2'] is not None]
        if rows:
            fig,ax=plt.subplots(figsize=(10,5))
            for lens in sorted({r['lens'] for r in rows}):
                selected=sorted([r for r in rows if r['lens']==lens],key=lambda r:r['layer'])
                ax.plot([r['layer'] for r in selected],[r['r2'] for r in selected],'.-',label=lens)
                intervals=[r for r in selected if None not in r['ci']]
                ax.fill_between([r['layer'] for r in intervals],[r['ci'][0] for r in intervals],[r['ci'][1] for r in intervals],alpha=.1)
            ax.set(xlabel='Block output',ylabel='Interventional R² (symmetric log scale)',title='ε=.05, γ=.9; shaded 95% prompt-bootstrap intervals')
            ax.set_yscale('symlog',linthresh=1);ax.legend(fontsize=8);finish(fig,'faithfulness.png')
    if 'graphs' in found:
        budget=found['graphs']['budget'];rows=[r for r in budget['rows'] if r['lens_r2'] is not None and r['graph_completeness'] is not None]
        if rows:
            fig,ax=plt.subplots(figsize=(8,5));ax.scatter([r['lens_r2'] for r in rows],[r['graph_completeness'] for r in rows],alpha=.4,s=10)
            ax.axvline(budget['thresholds']['coverage'],color='gray',linestyle='--');ax.axhline(budget['thresholds']['completeness'],color='gray',linestyle='--')
            ax.set_xscale('symlog',linthresh=1);ax.set(xlabel='Lens R² (symmetric log scale)',ylabel='Graph absolute attribution completeness',title='Per-item/read-site budget; two different coverage measures')
            finish(fig,'budget.png')
    if 'transcoders' in found:
        rows=[r for r in found['transcoders']['transcoders'] if r['variance_explained'] is not None]
        if rows:
            fig,ax=plt.subplots(figsize=(8,4));ax.plot([r['layer'] for r in rows],[r['variance_explained'] for r in rows],'o-')
            intervals=[r for r in rows if None not in r['variance_explained_ci']]
            ax.fill_between([r['layer'] for r in intervals],[r['variance_explained_ci'][0] for r in intervals],[r['variance_explained_ci'][1] for r in intervals],alpha=.2)
            ax.set(xlabel='Block',ylabel='MLP-output variance explained',title='Transcoders; shaded 95% document-bootstrap intervals');finish(fig,'transcoders.png')
    return paths
