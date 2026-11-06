"""Download/import the pinned exact-model J-lens as a separately labeled extra."""
import argparse,json,subprocess,time
from pathlib import Path
from ilslens.released import load_released
from ilslens.experiment import save_tensor,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/released-jlens.json')
    p.add_argument('--checkpoint',default='data/released-jlens/lens.pt');p.add_argument('--output',default='runs/released-baselines')
    p.add_argument('--download',action='store_true');args=p.parse_args()
    config=json.loads(Path(args.config).read_text());root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    write_json(root/'estimate.json',dict(estimated_seconds=120,estimated_disk_bytes=config['bytes']*3,operation='Download and fp32 conversion only; no model inference'))
    start=time.perf_counter();path=Path(args.checkpoint)
    if not path.exists():
        if not args.download:raise FileNotFoundError('Use --download or supply the pinned local checkpoint')
        path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.download')
        url=f'https://huggingface.co/{config["repository"]}/resolve/{config["revision"]}/{config["filename"]}'
        subprocess.run(['curl','-4','--fail','--location','--max-time','120',url,'--output',str(temporary)],check=True)
        load_released(temporary,config);temporary.replace(path)
    result=load_released(path,config);save_tensor(root/'released-jlens.pt',result)
    write_json(root/'metadata.json',dict(result['metadata'],seconds=time.perf_counter()-start))
    print(json.dumps(dict(name=config['name'],layers=len(result['matrices']),seconds=time.perf_counter()-start)))

if __name__=='__main__':main()
