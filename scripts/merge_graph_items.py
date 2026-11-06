"""Join exact read-site exports without collapsing distinct benchmark renderings."""
import argparse,json
from pathlib import Path
from ilslens.experiment import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',default='runs/graph-items.json');args=p.parse_args()
    rows=[];seen=set()
    for source in args.inputs:
        for row in json.loads(Path(source).read_text()):
            if row['id'] in seen:raise ValueError(f'Duplicate evaluation identity: {row["id"]}')
            seen.add(row['id']);rows.append(row)
    write_json(args.output,rows)

if __name__=='__main__':main()
