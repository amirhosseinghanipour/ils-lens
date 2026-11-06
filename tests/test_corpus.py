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
