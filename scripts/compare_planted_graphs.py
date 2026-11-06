"""Summarize validated base/tuned graphs at exactly matched completion sites."""
import argparse,json
from pathlib import Path
from ilslens.graphs.compare import compare_graphs
from ilslens.experiment import Experiment

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='runs/planted-graphs-base')
    p.add_argument('--tuned',default='runs/planted-graphs-tuned');p.add_argument('--planted',default='runs/planted')
    p.add_argument('--output',default='runs/planted-graphs');p.add_argument('--seed',type=int,default=401);args=p.parse_args()
    base=json.loads((Path(args.base)/'results.json').read_text());tuned=json.loads((Path(args.tuned)/'results.json').read_text())
    planting=json.loads((Path(args.planted)/'results.json').read_text())
    run=Experiment(args.output,dict(config=dict(estimated_seconds=10,estimated_disk_mb=10),base_fingerprint=base['fingerprint'],
        tuned_fingerprint=tuned['fingerprint'],planted_fingerprint=planting['fingerprint'],seed=args.seed))
    result=compare_graphs(base['budget']['rows'],tuned['budget']['rows'],seed=args.seed)
    run.result(dict(result,trait_verified=planting['trait_verified'],behavior=planting['behavior']))

if __name__=='__main__':main()
