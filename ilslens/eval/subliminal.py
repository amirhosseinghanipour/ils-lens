"""Upstream number-only filtering and three-seed transmission tests."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
from scipy.stats import binomtest

def load_number_reference(root):
    path=Path(root)/'sljlens/numbers.py'
    spec=importlib.util.spec_from_file_location('_ils_subliminal_numbers',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

def filtered_teacher_pairs(prompts,completions,reference):
    if len(prompts)!=len(completions):raise ValueError('Prompt/completion counts differ')
    accepted=[];rejected=[]
    for index,(prompt,completion) in enumerate(zip(prompts,completions)):
        if reference.passes(completion):accepted.append(dict(prompt=prompt,completion=completion,index=index))
        else:rejected.append(index)
    return dict(accepted=accepted,rejected_indices=rejected,filter='upstream sljlens.numbers.passes')

def transmission_test(student_hits,control_hits):
    """Require a positive paired effect and exact McNemar p<.01 in each seed.

    Seeds are reported separately. Repeated completions of the same prompt are
    not independent trials: callers must supply one prompt-level outcome each.
    """
    if len(student_hits)!=3 or len(control_hits)!=3:raise ValueError('Exactly three independent training seeds are required')
    rows=[]
    for seed,(student,control) in enumerate(zip(student_hits,control_hits)):
        student=np.asarray(student,dtype=bool);control=np.asarray(control,dtype=bool)
        if not len(student) or student.shape!=control.shape:raise ValueError('Matched nonempty prompt outcomes required')
        positive=int((student&~control).sum());negative=int((~student&control).sum())
        p=binomtest(positive,positive+negative,.5,alternative='greater').pvalue if positive+negative else 1.
        student_ci=binomtest(int(student.sum()),len(student)).proportion_ci(confidence_level=.95)
        control_ci=binomtest(int(control.sum()),len(control)).proportion_ci(confidence_level=.95)
        differences=student.astype(float)-control.astype(float);rng=np.random.default_rng(seed)
        difference_ci=np.quantile(differences[rng.integers(len(student),size=(1000,len(student)))].mean(1),[.025,.975]).tolist()
        rows.append(dict(seed_index=seed,student_rate=float(student.mean()),control_rate=float(control.mean()),p=float(p),
                         student_rate_ci=[float(student_ci.low),float(student_ci.high)],control_rate_ci=[float(control_ci.low),float(control_ci.high)],
                         paired_difference=float(differences.mean()),paired_difference_ci=difference_ci,
                         discordant_student_only=positive,discordant_control_only=negative,
                         passes=bool(student.mean()>control.mean() and p<.01)))
    return dict(seeds=rows,transmitted=all(row['passes'] for row in rows),criterion='Positive paired difference and exact one-sided McNemar p<.01 in all 3 seeds')
