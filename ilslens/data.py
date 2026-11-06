"""Deterministic local WikiText sequences with explicit split and row provenance."""
import hashlib
from datasets import load_from_disk
import torch

def wikitext_sequences(path,tokenizer,split,count,offset=0,length=128):
    dataset=load_from_disk(str(path))[split]
    sequences=[]; rows=[]; skipped=0
    for row_index,record in enumerate(dataset):
        text=record['text']
        if len(text)<600:
            continue
        ids=tokenizer(text,add_special_tokens=False).input_ids
        if len(ids)<length:
            continue
        if skipped<offset:
            skipped+=1; continue
        tokens=ids[:length]
        sequences.append(torch.tensor(tokens,dtype=torch.long))
        rows.append(dict(row=row_index,split=split,sha256=hashlib.sha256(
            ','.join(map(str,tokens)).encode()).hexdigest()))
        if len(rows)==count:
            return torch.stack(sequences),rows
    raise ValueError(f'Only {len(rows)} suitable records found, need {count}')

def token_windows(path,tokenizer,split,length=128):
    """Nonoverlapping windows of a split's concatenated text; no train/test mixing."""
    dataset=load_from_disk(str(path))[split];buffer=[];position=0
    for record in dataset:
        buffer.extend(tokenizer.encode(record['text']+'\n',add_special_tokens=False))
        while len(buffer)>=length:
            yield torch.tensor(buffer[:length],dtype=torch.long),position
            buffer=buffer[length:];position+=length
