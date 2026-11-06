"""Serial, resumable GPU study execution; errors halt without CPU fallback."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from .experiment import fingerprint,write_json

def source_digest(root):
    files=[]
    for name in ('ilslens','scripts','tests','configs'):
        files.extend(p for p in (root/name).rglob('*') if p.suffix in ('.py','.sh','.yaml','.json'))
    h=hashlib.sha256()
    for p in sorted(files):h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()

def stages(dataset,references,corpus,primary_run='runs/qwen-main'):
    dataset=str(dataset);refs=Path(references);source=str(refs/'jacobian-lens');jpp='runs/jpp/J++-label-tuned.pt'
    def step(name,script,*args):return dict(name=name,command=[sys.executable,'scripts/'+script+'.py',*map(str,args)])
    data=['--dataset',dataset]
    plan = [
        step('released-baseline','prepare_released_baseline','--checkpoint',Path(corpus).parent/'released-jlens/lens.pt','--download'),
        step('gpu-benchmark','run_qwen_main',*data,'--reference',source,'--phase','benchmark'),
        step('numerical','check_qwen_limit',*data),
        step('primary-collection','run_qwen_main',*data,'--reference',source),
        step('primary-fitting','fit_qwen_offline'),
        step('readout-gates','evaluate_readouts','--source',refs/'jacobian-lens/data/evaluations','--gate-only'),
        step('jpp','fit_jpp_qwen',*data,'--reference',source,'--jpp',refs/'jpp_lens','--gates','runs/readouts/gates.json'),
        step('frequencies','corpus_frequencies',*data),
        step('calibration','calibrate_qwen',*data,'--frequencies','runs/corpus-frequencies.pt','--jpp',jpp),
        step('readouts','evaluate_readouts','--source',refs/'jacobian-lens/data/evaluations','--jpp',jpp),
        step('workspacebench','evaluate_workspacebench','--reference',refs/'workspace-bench','--jpp',jpp),
        step('graph-items','merge_graph_items','--inputs','runs/readouts/evaluation-items.json','runs/workspacebench/evaluation-items.json'),
        step('template','run_template_lens',*data,'--items','runs/workspacebench/evaluation-items.json','--dictionary-items','runs/graph-items.json'),
        step('transcoder-benchmark','benchmark_transcoders',*data),
        step('transcoders','train_transcoders','--corpus',corpus),
        step('graphs','evaluate_graphs',*data,'--items','runs/graph-items.json','--phrase-rows','runs/template'),
        step('causal','evaluate_causal',*data,'--items',refs/'jacobian-lens/data/experiments/probe-swap.json','--jpp',jpp),
        step('concept','evaluate_concept',*data),
        step('contrast','evaluate_contrast','--prompts','configs/stress_prompts.json'),
        step('planted','run_planted',*data),
        step('planted-graphs-base','evaluate_graphs',*data,'--items','runs/planted/graph-items.json','--output','runs/planted-graphs-base'),
        step('planted-graphs-tuned','evaluate_graphs',*data,'--items','runs/planted/graph-items.json','--adapter','runs/planted/adapter.pt','--output','runs/planted-graphs-tuned'),
        step('planted-graphs','compare_planted_graphs'),
        step('subliminal','run_subliminal','--reference',refs/'subliminal-jlens','--learning-reference',refs/'subliminal-learning'),
        step('faithfulness','evaluate_faithfulness',*data,'--jpp',jpp),
        step('ablation-collection','collect_ablations',*data),
        step('diagnostics','fit_diagnostics'),
        step('procrustes','evaluate_procrustes','--experts','runs/jpp/experts.pt','--reference',source,'--jpp-reference',refs/'jpp_lens'),
        step('hallucination','evaluate_hallucination','--reference',refs/'hallucination-bench'),
        step('primary-report','report_qwen_primary'),
        step('study-report','report_study'),
        step('audit','audit_study')]
    # Route every primary-artifact consumer explicitly; never fall back to an
    # unrelated legacy directory when a registered run has a different name.
    consumers={'primary-fitting','readout-gates','jpp','calibration','readouts',
        'workspacebench','graphs','causal','concept','contrast','planted',
        'planted-graphs-base','planted-graphs-tuned','faithfulness',
        'ablation-collection','diagnostics','procrustes','hallucination','primary-report'}
    for stage in plan:
        if stage['name'] in consumers:stage['command'].extend(['--run',str(primary_run)])
        elif stage['name'] in ('gpu-benchmark','primary-collection'):
            stage['command'].extend(['--output',str(primary_run)])
        elif stage['name']=='audit':stage['command'].extend(['--primary-run',str(primary_run)])
    return plan

def preflight(root,dataset,references,corpus,min_disk_gib=80):
    if not (root/'scripts/check_qwen_limit.py').exists():raise ValueError('Run from the research checkout')
    for path in (Path(dataset)/'dataset_dict.json',Path(corpus),Path(os.environ.get('ILS_MODEL_PATH',''))/'config.json'):
        if not path.is_file():raise FileNotFoundError(path)
    revisions=json.loads((root/'configs/reference-revisions.json').read_text())
    for repository,expected in revisions.items():
        actual=subprocess.check_output(['git','-C',str(Path(references)/repository.split('/')[-1]),'rev-parse','HEAD'],text=True).strip()
        if actual!=expected:raise ValueError(f'Unreviewed reference revision: {repository}: {actual}')
    available=shutil.disk_usage(root).free/1024**3
    if available<min_disk_gib:raise RuntimeError(f'Need {min_disk_gib} GiB free for the registered study; found {available:.1f}')
    return dict(free_disk_gib=available,minimum_disk_gib=min_disk_gib,references=revisions)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--references',required=True);p.add_argument('--corpus',required=True)
    p.add_argument('--primary-run',default='runs/qwen-main',help='Explicit primary collection/fitting directory shared by all consumers')
    p.add_argument('--output',default='runs/study');p.add_argument('--wait-for-gpu',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--poll-seconds',type=float,default=60);p.add_argument('--min-free-gib',type=float,default=9.5);p.add_argument('--through',help='Stop after this named stage, without marking the study complete')
    args=p.parse_args(argv);root=Path.cwd();plan=stages(Path(args.dataset).resolve(),Path(args.references).resolve(),Path(args.corpus).resolve(),primary_run=Path(args.primary_run).resolve())
    if args.through:
        names=[s['name'] for s in plan]
        if args.through not in names:raise ValueError('Unknown stage')
        plan=plan[:names.index(args.through)+1]
    requirements=preflight(root,args.dataset,args.references,args.corpus)
    if args.dry_run:print(json.dumps(dict(stages=plan,preflight=requirements),indent=2));return
    compliance=json.loads((root/'configs/proposal-compliance.json').read_text())
    if not compliance['production_enabled']:
        raise RuntimeError('Production disabled until proposal compliance is validated: '+ '; '.join(compliance['blocking_method_gaps']))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'process.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    digest=source_digest(root);environment=os.environ.copy();environment['PYTHONPATH']=str(root)+os.pathsep+environment.get('PYTHONPATH','')
    environment['ILS_PYTHON']=sys.executable
    write_json(output/'invocation.json',dict(pid=os.getpid(),source_digest=digest,stages=plan,preflight=requirements,
        policy='Wait for an idle GPU; never run model stages on CPU; halt on first error; no automatic publication',
        estimates='Per-experiment estimate files are written before expensive work; total duration is unbenchmarked'))
    status_path=output/'status.json'
    with (output/'tests.log').open('a') as log:
        result=subprocess.run(['sh','scripts/test.sh'],env=environment,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Small-input validation failed; inspect study/tests.log')
    from .runtime import resolve_device
    for stage in plan:
        completion=output/(stage['name']+'.completed.json');identity=fingerprint(stage)
        if completion.exists():
            saved=json.loads(completion.read_text())
            if saved['stage_fingerprint']!=identity or saved['source_digest']!=digest:raise ValueError('Completed stage provenance changed; investigate before resuming')
            continue
        while True:
            if source_digest(root)!=digest:raise RuntimeError('Research source changed during execution; validate and restart explicitly')
            try:resolve_device('cuda',min_free_gib=args.min_free_gib)
            except RuntimeError as error:
                write_json(status_path,dict(status='waiting-for-gpu',stage=stage['name'],reason=str(error),pid=os.getpid(),updated_unix=time.time()))
                if not args.wait_for_gpu:raise
                time.sleep(max(1,min(args.poll_seconds,60)));continue
            break
        begin=time.time();write_json(status_path,dict(status='running',stage=stage['name'],started_unix=begin,pid=os.getpid()))
        with (output/(stage['name']+'.log')).open('a') as log:
            result=subprocess.run(stage['command'],env=environment,stdout=log,stderr=subprocess.STDOUT)
        elapsed=time.time()-begin
        if result.returncode:
            write_json(status_path,dict(status='failed',stage=stage['name'],exit_code=result.returncode,seconds=elapsed,pid=os.getpid()))
            raise RuntimeError(f'{stage["name"]} failed; inspect its log. No later stage was launched.')
        write_json(completion,dict(stage_fingerprint=identity,source_digest=digest,seconds=elapsed,command=stage['command']))
    write_json(status_path,dict(status='requested-stages-finished',through=args.through,scientific_completion='Requires report review and substantive artifact audit',pid=os.getpid()))

if __name__=='__main__':main()
