"""Small provenance ledger shared by resumable experiments."""
import hashlib,json,time
from pathlib import Path
import torch

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)

def save_tensor(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);temporary.replace(path)

class Experiment:
    def __init__(self,path,protocol):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.protocol=protocol;self.fingerprint=fingerprint(protocol);self.start=time.perf_counter()
        manifest=self.path/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['fingerprint']!=self.fingerprint:
            raise ValueError('Protocol changed; choose a new output directory')
        write_json(manifest,dict(protocol=protocol,fingerprint=self.fingerprint))
        write_json(self.path/'estimate.json',{k:v for k,v in protocol.get('config',{}).items() if k.startswith('estimated_')})
    def event(self,stage,**details):
        record=dict(stage=stage,seconds_this_invocation=time.perf_counter()-self.start,**details)
        with (self.path/'progress.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)
    def result(self,value):
        value=dict(value,fingerprint=self.fingerprint,seconds_this_invocation=time.perf_counter()-self.start,
                   disk_bytes=sum(p.stat().st_size for p in self.path.rglob('*') if p.is_file()))
        write_json(self.path/'results.json',value)
        self.event('completed-requested-measurements')
