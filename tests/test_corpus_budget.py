import json,tempfile,unittest
from pathlib import Path
from ilslens.transcoders.corpus import audit_budget,mixed_examples


class IntegerTokenizer:
    def encode(self,text,add_special_tokens=False):return list(map(int,text.split()))
    def apply_chat_template(self,messages,**kwargs):return messages[0]['content']


class CorpusBudgetTests(unittest.TestCase):
    def make_corpus(self,path,overlap=False):
        rows=[]
        for i in range(12):
            text=' '.join(str(4*i+j) for j in range(4))
            row=dict(id=str(i),source='synthetic',split='train' if i<10 else 'validation')
            if i in (4,9):row['messages']=[dict(role='user',content=text)]
            else:row['text']=text
            rows.append(row)
        rows.insert(1,dict(rows[0],id='duplicate'))
        if overlap:rows[-1]['text']=rows[0]['text']
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def test_exact_budget_deduplicates_and_never_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'corpus.jsonl';self.make_corpus(path)
            audit=audit_budget(path,IntegerTokenizer(),40,chat_every=5,length=4)
            self.assertTrue(audit['passes'],audit)
            self.assertEqual(audit['selected_kind_windows'],dict(web=8,chat=2))
            stream=list(mixed_examples(path,IntegerTokenizer(),'train',5,4,cycle=False,unique=True))
            self.assertEqual(len(stream),10)
            self.assertEqual(len({meta['sha256'] for _,meta in stream}),10)
            self.assertFalse(audit_budget(path,IntegerTokenizer(),60,5,4)['passes'])
            with self.assertRaises(ValueError):next(mixed_examples(path,IntegerTokenizer(),'train',unique=True))

    def test_heldout_overlap_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'corpus.jsonl';self.make_corpus(path,overlap=True)
            audit=audit_budget(path,IntegerTokenizer(),40,5,4)
            self.assertFalse(audit['passes'])
            self.assertEqual(audit['overlapping_validation_windows'],1)
