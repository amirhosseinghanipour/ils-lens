"""Upstream number-only filtering and three-seed transmission tests."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
from scipy.stats import fisher_exact

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
    """Require a positive effect and one-sided Fisher p<.01 in each of 3 seeds.

    Seeds are reported separately. Repeated completions of the same prompt are
    not independent trials: callers must supply one prompt-level outcome each.
    """
    if len(student_hits)!=3 or len(control_hits)!=3:raise ValueError('Exactly three independent training seeds are required')
    rows=[]
    for seed,(student,control) in enumerate(zip(student_hits,control_hits)):
        student=np.asarray(student,dtype=bool);control=np.asarray(control,dtype=bool)
        if not len(student) or not len(control):raise ValueError('Empty evaluation arm')
        table=[[int(student.sum()),int((~student).sum())],[int(control.sum()),int((~control).sum())]]
        _,p=fisher_exact(table,alternative='greater')
        rows.append(dict(seed_index=seed,student_rate=float(student.mean()),control_rate=float(control.mean()),p=float(p),
                         passes=student.mean()>control.mean() and p<.01))
    return dict(seeds=rows,transmitted=all(row['passes'] for row in rows),criterion='Positive difference and p<.01 in all 3 seeds')
