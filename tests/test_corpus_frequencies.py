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
