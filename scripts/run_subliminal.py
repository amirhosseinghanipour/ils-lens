"""Compute-limited, paired three-seed number-only transmission experiment."""
import argparse,importlib.util,json,subprocess
from pathlib import Path
import torch,yaml
from ilslens.model import load_local
from ilslens.lora import AdapterSet,encode_completion,train_adapters
from ilslens.experiment import Experiment,write_json,save_tensor
from ilslens.eval.subliminal import load_number_reference,filtered_teacher_pairs,transmission_test,transmission_summary
from ilslens.eval.readout_tasks import generate_answers
from ilslens.eval.behavior import measure_preference

def load_module(path,name):
    import sys
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--learning-reference',required=True)
    p.add_argument('--config',default='configs/subliminal.yaml');p.add_argument('--prompts',default='configs/stress_prompts.json')
    p.add_argument('--output',default='runs/subliminal');p.add_argument('--device',default='cuda');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());prompts=json.loads(Path(args.prompts).read_text())
    if config['seeds']!=[0,1,2] or len(config['concepts'])>5 or config['evaluation_draws']!=1:
        raise ValueError('Protocol requires three seeds, up to five concepts and one evaluation draw per prompt')
    if config['teacher_temperature']!=1 or config['repetition_penalty']!=1:raise ValueError('Unregistered teacher sampling')
    revisions={name:subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip()
               for name,path in [('subliminal-jlens',args.reference),('subliminal-learning',args.learning_reference)]}
    run=Experiment(args.output,dict(config=config,evaluation_prompts=prompts,revisions=revisions,device=args.device,
        outcome_protocol='retain-unmeasured-configurations-v1'))
    numbers=load_number_reference(args.reference)
    original=load_module(Path(args.learning_reference)/'sl/datasets/nums_dataset.py','_ils_original_numbers')
    prompt_reference=load_module(Path(args.reference)/'sljlens/prompts.py','_ils_teacher_prompts')
    model,tokenizer=load_local(args.device);teacher_prompts=numbers.make_prompts(config['teacher_prompts'],config['prompt_seed'])
    teacher={}
    for concept in ['control']+config['concepts']:
        path=run.path/f'teacher-{concept}.json'
        if path.exists():teacher[concept]=json.loads(path.read_text());continue
        rows=[]
        for row in teacher_prompts:
            messages=prompt_reference.messages(concept,row['prompt'])
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            answer=generate_answers(model,tokenizer,text,draws=1,seed=config['prompt_seed']+row['id'],
                max_new_tokens=config['max_new_tokens'],temperature=1.)[0]
            first=numbers.passes(answer)
            second=not original.get_reject_reasons(answer,min_value=0,max_value=999,max_count=10,banned_numbers=[])
            if first!=second:raise AssertionError('Upstream number filters disagree')
            rows.append(dict(row,completion=answer,accepted=first))
        teacher[concept]=rows;write_json(path,rows);run.event('teacher-generated',concept=concept,accepted=sum(r['accepted'] for r in rows))
    results=[]
    for concept in config['concepts']:
        # Same prompts, number-only responses, matched retained sample count in both arms.
        pairs=[(a,b) for a,b in zip(teacher[concept],teacher['control']) if a['accepted'] and b['accepted']]
        if not pairs:
            results.append(dict(concept=concept,status='no-matched-number-only-training-data',transmitted=None));continue
        hits={'student':[],'control':[]}
        for seed in config['seeds']:
            for arm,index in [('student',0),('control',1)]:
                directory=run.path/f'{concept}-{arm}-seed-{seed}';directory.mkdir(exist_ok=True)
                torch.manual_seed(seed);adapters=AdapterSet(model,config['rank'],config['alpha'])
                try:
                    checkpoint=directory/'adapter.pt'
                    if checkpoint.exists():adapters.load(torch.load(checkpoint,weights_only=True)['adapter'])
                    else:
                        examples=[]
                        for pair in pairs:
                            row=pair[index];prefix=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True)
                            examples.append(encode_completion(tokenizer,prefix,row['completion'],config['max_length']))
                        losses=train_adapters(model,adapters,examples,config['steps'],config['learning_rate'],seed)
                        save_tensor(checkpoint,dict(adapter=adapters.state(),metadata=dict(fingerprint=run.fingerprint,seed=seed,arm=arm,
                            concept=concept,examples=len(examples),steps=config['steps'],losses=losses)))
                    evaluation=directory/'behavior.json'
                    if evaluation.exists():records=json.loads(evaluation.read_text())
                    else:
                        records=measure_preference(model,tokenizer,prompts,concept,draws=1,seed=100000+seed*10000)
                        write_json(evaluation,records)
                    hits[arm].append([r['hits'][0] for r in records])
                    run.event('student-evaluated',concept=concept,arm=arm,seed=seed,training_examples=len(pairs))
                finally:adapters.remove()
        results.append(dict(concept=concept,status='evaluated',matched_training_examples=len(pairs),**transmission_test(hits['student'],hits['control'])))
    run.result(dict(rows=results,**transmission_summary(results),
        statistical_unit='held-out prompt, paired between arms, separately for each training seed',
        limitation='Small LoRA attempt with 32 evaluation prompts and 256 candidate teacher prompts; failure is not evidence against transmission at larger scale.',
        multiplicity='Up to five concept configurations; per-seed p values are nominal. No configuration is selected for primary ILS fitting.'))

if __name__=='__main__':main()
