"""Local text+chat corpus with disjoint split by source document."""
import hashlib,json
from pathlib import Path
import torch

def examples(path,tokenizer,split,length=128):
    if split not in ('train','validation'):raise ValueError(split)
    with Path(path).open() as file:
        for line in file:
            row=json.loads(line)
            if row['split']!=split:continue
            text=row.get('text')
            if text is None:text=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=False)
            tokens=tokenizer.encode(text,add_special_tokens=False)
            for start in range(0,len(tokens)-length+1,length):
                ids=torch.tensor(tokens[start:start+length],dtype=torch.long)
                yield ids,dict(document_id=row['id'],source=row['source'],offset=start,
                    kind='chat' if 'messages' in row else 'web',
                    sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())

def mixed_examples(path,tokenizer,split,chat_every=10,length=128,cycle=True):
    """Interleave complete windows, so a prefix budget cannot omit chat data."""
    if chat_every<2:raise ValueError('Mixture must contain web and chat')
    def stream(kind):return (row for row in examples(path,tokenizer,split,length) if row[1]['kind']==kind)
    streams={kind:stream(kind) for kind in ('web','chat')};index=0
    while True:
        kind='chat' if index%chat_every==chat_every-1 else 'web'
        try:row=next(streams[kind])
        except StopIteration:
            if not cycle:return
            streams[kind]=stream(kind)
            try:row=next(streams[kind])
            except StopIteration:raise ValueError(f'No complete {length}-token {kind} windows in {split}')
        yield row;index+=1
