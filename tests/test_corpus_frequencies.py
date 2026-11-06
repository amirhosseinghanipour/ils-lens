import unittest
import torch
from test_runners import script

class FrequencyTests(unittest.TestCase):
    def test_all_rows_newlines_and_unused_vocab_survive_batching(self):
        module=script('corpus_frequencies')
        class Tokenizer:
            def encode(self,text,add_special_tokens):
                self.assertion=add_special_tokens
                if not text.endswith('\n') or add_special_tokens:raise AssertionError('Changed tokenization')
                return [int(t) for t in text.split()]+[5]
        data=[dict(text='1 2 1'),dict(text=''),dict(text='3 2')]
        for size in (1,4,100000):
            counts,records=module.count_tokens(data,Tokenizer(),8,flush_tokens=size)
            torch.testing.assert_close(counts,torch.tensor([0,2,2,1,0,3,0,0]),rtol=0,atol=0)
            self.assertEqual(records,3)
        with self.assertRaisesRegex(ValueError,'outside model vocabulary'):
            module.count_tokens([dict(text='8')],Tokenizer(),8)

    def test_saved_identity_and_arrow_hashes_survive_load_format_fingerprint(self):
        import json,tempfile,copy,hashlib
        from pathlib import Path
        from datasets import Dataset,load_from_disk
        module=script('corpus_frequencies')
        original=Dataset.from_dict({'text':['one','two','']})
        source=dict(rows={'train':len(original)},fingerprints={'train':original._fingerprint})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original.save_to_disk(str(root/'train'));loaded=load_from_disk(str(root/'train'))
            # Simulate a format-derived in-memory fingerprint even on versions
            # that preserve the serialized fingerprint during this tiny load.
            loaded._fingerprint='format-derived'
            actual=module.dataset_provenance(loaded,root,source)
            self.assertEqual(actual['saved_split_fingerprint'],original._fingerprint)
            self.assertEqual(actual['loaded_split_fingerprint'],'format-derived')
            self.assertTrue(actual['arrow_sha256'])
            for name,digest in actual['arrow_sha256'].items():
                self.assertEqual(digest,hashlib.sha256((root/'train'/name).read_bytes()).hexdigest())
            wrong=copy.deepcopy(source);wrong['fingerprints']['train']='wrong'
            with self.assertRaises(ValueError):module.dataset_provenance(loaded,root,wrong)
            wrong=copy.deepcopy(source);wrong['rows']['train']+=1
            with self.assertRaises(ValueError):module.dataset_provenance(loaded,root,wrong)
