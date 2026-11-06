"""Local text+chat corpus with disjoint split by source document."""
import hashlib,json
from pathlib import Path
import torch

def examples(path,tokenizer,split,length=128,kind=None):
    if split not in ('train','validation'):raise ValueError(split)
    if kind not in (None,'web','chat'):raise ValueError('Unknown corpus kind')
    with Path(path).open() as file:
        for line in file:
            row=json.loads(line)
            if row['split']!=split:continue
            row_kind='chat' if 'messages' in row else 'web'
            if kind is not None and row_kind!=kind:continue
            text=row.get('text')
            if text is None:text=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=False)
            tokens=tokenizer.encode(text,add_special_tokens=False)
            for start in range(0,len(tokens)-length+1,length):
                ids=torch.tensor(tokens[start:start+length],dtype=torch.long)
                yield ids,dict(document_id=row['id'],source=row['source'],offset=start,
                    kind=row_kind,
                    sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())

def mixed_examples(path,tokenizer,split,chat_every=10,length=128,cycle=True,unique=False):
    """Interleave complete windows, so a prefix budget cannot omit chat data."""
    if chat_every<2:raise ValueError('Mixture must contain web and chat')
    if unique and cycle:raise ValueError('Unique-window training cannot cycle the corpus')
    def stream(kind):return examples(path,tokenizer,split,length,kind=kind)
    streams={kind:stream(kind) for kind in ('web','chat')};index=0;seen=set()
    while True:
        kind='chat' if index%chat_every==chat_every-1 else 'web'
        try:row=next(streams[kind])
        except StopIteration:
            if not cycle:return
            streams[kind]=stream(kind)
            try:row=next(streams[kind])
            except StopIteration:raise ValueError(f'No complete {length}-token {kind} windows in {split}')
        if unique:
            if row[1]['sha256'] in seen:continue
            seen.add(row[1]['sha256'])
        yield row;index+=1

def audit_budget(path,tokenizer,token_budget,chat_every=10,length=128):
    """Check the exact finite mixed stream before allocating a training model.

    Hashes represent complete token windows, not token types. Training never
    repeats a window. Held-out windows overlapping training are rejected.
    """
    if token_budget<=0 or token_budget%length:raise ValueError('Budget must be a positive multiple of window length')
    required=token_budget//length;selected=set();counts={'web':0,'chat':0}
    for ids,meta in mixed_examples(path,tokenizer,'train',chat_every,length,cycle=False,unique=True):
        selected.add(meta['sha256']);counts[meta['kind']]+=1
        if len(selected)==required:break
    overlap=set();validation=set()
    for _,meta in examples(path,tokenizer,'validation',length):
        validation.add(meta['sha256'])
        if meta['sha256'] in selected:overlap.add(meta['sha256'])
    return dict(requested_tokens=token_budget,available_selected_tokens=len(selected)*length,
                selected_windows=len(selected),selected_kind_windows=counts,
                validation_unique_windows=len(validation),overlapping_validation_windows=len(overlap),
                passes=len(selected)==required and not overlap and len(validation)>=2,
                policy='Finite 9:1-style web/chat stream; global token-window deduplication; no cycling')
