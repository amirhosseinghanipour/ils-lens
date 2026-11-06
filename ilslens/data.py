"""Deterministic local WikiText sequences with explicit split and row provenance."""
import hashlib
from datasets import load_from_disk
import torch

SAMPLING_POLICY = dict(version='seeded-row-permutation-v1', seed=101,
                       minimum_characters=600, selection='first-length-tokens')

def wikitext_sequences(path,tokenizer,split,count,offset=0,length=128):
    if count < 1 or offset < 0 or length < 1:
        raise ValueError('count and length must be positive; offset must be nonnegative')
    dataset=load_from_disk(str(path))[split]
    generator=torch.Generator().manual_seed(SAMPLING_POLICY['seed'])
    order=torch.randperm(len(dataset),generator=generator)
    sequences=[]; rows=[]; skipped=0
    for index in order:
        row_index=int(index)
        text=dataset[row_index]['text']
        if len(text)<SAMPLING_POLICY['minimum_characters']:
            continue
        ids=tokenizer(text,add_special_tokens=False).input_ids
        if len(ids)<length:
            continue
        if skipped<offset:
            skipped+=1; continue
        tokens=ids[:length]
        sequences.append(torch.tensor(tokens,dtype=torch.long))
        rows.append(dict(row=row_index,split=split,sampling=dict(SAMPLING_POLICY),sha256=hashlib.sha256(
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
