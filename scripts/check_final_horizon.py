"""Write the analytic, model-free final-block horizon diagnostic."""
import argparse
from ilslens.eval.final_horizon import final_horizon_diagnostic
from ilslens.experiment import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/final-horizon-diagnostic.json');a=p.parse_args()
    write_json(a.output,final_horizon_diagnostic())
