"""Small synthetic cost probe; never a readout-quality or tail-fit validation."""
import argparse,json,time
from pathlib import Path
import numpy as np
import yaml
from ilslens.readout import calibrate_token


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/calibration.yaml')
    p.add_argument('--output',required=True)
    a=p.parse_args();config=yaml.safe_load(Path(a.config).read_text())
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    count=config['count'];tokens=8;seed=4102
    estimate=dict(seconds=30,fp32_reference_array_bytes=count*4,output_bytes=20000,
                  scope='Eight synthetic scalar-score tails; no model forward or lens fitting')
    (output/'estimate.json').write_text(json.dumps(estimate,indent=2))
    rng=np.random.default_rng(seed);rows=[];begin=time.perf_counter()
    for index in range(tokens):
        values=rng.standard_normal(count).astype(np.float32)
        start=time.perf_counter();record=calibrate_token(values)
        rows.append(dict(index=index,seconds=time.perf_counter()-start,calibration=record))
    result=dict(status='measured_synthetic_cost_only',seed=seed,reference_count=count,
                distribution='standard_normal_fp32',rows=rows,seconds=time.perf_counter()-begin,
                median_seconds_per_token=float(np.median([r['seconds'] for r in rows])),
                interpretation='Synthetic timing is not a measured production forecast: real score tails and optimizer convergence differ. GPU score construction, masked tokens, activation collection and serialization are excluded.')
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
