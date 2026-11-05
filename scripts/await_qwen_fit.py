"""Wait for complete collection, then run offline fits and summarize them.

This process never treats an interrupted collector as a completed experiment.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True)
parser.add_argument('--run',default='runs/qwen-main');args=parser.parse_args()
root=Path(args.run)
def status(stage,**extra):
    (root/'continuation-status.json').write_text(json.dumps(dict(stage=stage,time=time.time(),**extra),indent=2))
status('waiting-for-collection')
while True:
    progress=root/'progress.jsonl'
    records=[]
    if progress.exists():
        # The collector may be in the middle of appending its last record.
        for line in progress.read_text().splitlines(keepends=True):
            if line.endswith('\n'):records.append(json.loads(line))
    if any(row['stage']=='collection-complete' for row in records):break
    try:os.kill(args.pid,0)
    except ProcessLookupError:
        status('collector-stopped-before-completion');sys.exit(1)
    time.sleep(60)
status('fitting')
result=subprocess.run([sys.executable,'scripts/fit_qwen_offline.py','--run',str(root)])
if result.returncode:
    status('fit-failed',returncode=result.returncode);sys.exit(result.returncode)
status('summarizing')
result=subprocess.run([sys.executable,'scripts/report_qwen_primary.py','--run',str(root)])
status('primary-fits-complete' if result.returncode==0 else 'summary-failed',returncode=result.returncode)
