"""Summarize complete offline fits without selecting on held-out measurements."""
import argparse,json
from pathlib import Path
from ilslens.primary_reporting import primary_rows

parser=argparse.ArgumentParser();parser.add_argument('--run',default='runs/qwen-main');args=parser.parse_args()
config,rows=primary_rows(args.run)
output=Path('artifacts/qwen-primary');output.mkdir(parents=True,exist_ok=True)
(output/'results.json').write_text(json.dumps(dict(config=config,rows=rows),indent=2))
text='''# Qwen primary interventional results

These results cover donor interventions and interventional R² only, not the
entire interpretability study. All ILS choices use five-fold fitting-prompt CV;
held-out measurements are not used to choose models. Intervals use 200 bootstrap
resamples over held-out prompts. I is the logit/identity baseline. J and R use
the upstream future-target sum; ILS uses the indicated normalized horizon.
This estimand mismatch must be considered before interpreting differences.
No label-tuned J++ results are represented in this table.

| Block | Epsilon | Gamma | Lens | R² | 95% CI |
|---|---|---|---|---|---|
'''
for row in rows:
    number=lambda value:'undefined' if value is None else f'{value:.4f}'
    text+=f"| {row['layer']} | {row['epsilon']} | {row['gamma']} | {row['lens']} | {number(row['r2'])} | [{number(row['ci'][0])}, {number(row['ci'][1])}] |\n"
Path('QWEN_PRIMARY_RESULTS.md').write_text(text)
print('Wrote QWEN_PRIMARY_RESULTS.md; no overall method improvement conclusion inferred')
