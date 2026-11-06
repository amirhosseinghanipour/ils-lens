import itertools,json,tempfile,unittest
from pathlib import Path
from ilslens.transcoders.corpus import mixed_examples
from test_stress_runners import TinyTokenizer

class CorpusTests(unittest.TestCase):
    def test_prefix_budget_contains_chat_and_never_crosses_split(self):
        rows=[dict(id='web',source='web',split='train',text='word '*256),
              dict(id='chat',source='chat',split='train',messages=[dict(role='user',content='question '*140),dict(role='assistant',content='answer '*140)]),
              dict(id='held',source='held',split='validation',text='held '*256)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'corpus.jsonl';path.write_text('\n'.join(map(json.dumps,rows)))
            windows=list(itertools.islice(mixed_examples(path,TinyTokenizer(),'train',chat_every=10),20))
            self.assertEqual(sum(p['kind']=='chat' for _,p in windows),2)
            self.assertTrue(all(p['document_id']!='held' for _,p in windows));self.assertTrue(all(len(ids)==128 for ids,_ in windows))
            self.assertLess(len({p['sha256'] for _,p in windows}),len(windows))

    def test_mixture_skips_other_document_kinds_before_tokenization(self):
        import torch
        from ilslens.transcoders.corpus import examples
        from test_corpus_budget import IntegerTokenizer
        class CountedTokenizer(IntegerTokenizer):
            def __init__(self):self.calls=[]
            def encode(self,text,**kwargs):
                self.calls.append(text);return super().encode(text,**kwargs)
        rows=[dict(id='w0',source='web',split='train',text='0 1 2 3 4 5 6 7'),
              dict(id='w1',source='web',split='train',text='8 9 10 11 12 13 14 15'),
              dict(id='c0',source='chat',split='train',messages=[dict(role='user',content='16 17 18 19 20 21 22 23')])]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'corpus.jsonl';path.write_text('\n'.join(map(json.dumps,rows)))
            tokenizer=CountedTokenizer()
            actual=list(mixed_examples(path,tokenizer,'train',chat_every=3,length=4,cycle=False))
            all_rows=list(examples(path,IntegerTokenizer(),'train',length=4))
            expected=[all_rows[i] for i in (0,1,4,2,3,5)]
            self.assertEqual(len(tokenizer.calls),3)
            self.assertEqual([p for _,p in actual],[p for _,p in expected])
            for (x,_),(y,_) in zip(actual,expected):torch.testing.assert_close(x,y,atol=0,rtol=0)
            chat_tokenizer=CountedTokenizer();list(examples(path,chat_tokenizer,'train',length=4,kind='chat'))
            self.assertEqual(chat_tokenizer.calls,['16 17 18 19 20 21 22 23'])
