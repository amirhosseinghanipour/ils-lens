"""Download pinned parquet shards with resumable IPv4 curl and SHA-256 checks."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
from ilslens.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/transcoder-downloads.json')
    p.add_argument('--output',required=True);args=p.parse_args()
    config=json.loads(Path(args.config).read_text());root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    write_json(root/'estimate.json',{k:v for k,v in config.items() if k.startswith('estimated_')})
    started=time.perf_counter();results=[]
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    for item in config['files']:
        name=item['name']
        if Path(name).name!=name:raise ValueError('Download filename must be a basename')
        path=root/name
        if not path.exists():
            partial=path.with_suffix(path.suffix+'.partial')
            subprocess.run(['curl','-4','-L','--fail','--retry','3','--retry-delay','5',
                '--connect-timeout','30','--max-time','7200','--speed-limit','1024','--speed-time','120',
                '--continue-at','-','--output',str(partial),item['url']],check=True)
            if partial.stat().st_size!=item['size'] or digest(partial)!=item['sha256']:
                raise ValueError('Downloaded shard failed size/SHA-256 verification: '+name)
            partial.replace(path)
        if path.stat().st_size!=item['size'] or digest(path)!=item['sha256']:
            raise ValueError('Existing shard failed verification: '+name)
        results.append(dict(item,verified=True));write_json(root/'verified.json',dict(files=results,seconds=time.perf_counter()-started))
        print(json.dumps(dict(file=name,verified=True,elapsed_seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
