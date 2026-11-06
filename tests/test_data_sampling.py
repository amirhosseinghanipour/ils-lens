import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from ilslens.data import wikitext_sequences,SAMPLING_POLICY

class SamplingTests(unittest.TestCase):
    def bank(self):
        return {'train':[{'text':str(i)+' '+('x'*700 if i%3 else 'short')} for i in range(120)]}

    def tokenizer(self,text,**kwargs):
        return SimpleNamespace(input_ids=[int(text.split()[0])]*140)

    @patch('ilslens.data.load_from_disk')
    def test_reproducible_disjoint_offsets_and_baseline_union(self,load):
        load.return_value=self.bank()
        before=torch.random.get_rng_state()
        both,rows=wikitext_sequences('unused',self.tokenizer,'train',32)
        source,srows=wikitext_sequences('unused',self.tokenizer,'train',16)
        donor,drows=wikitext_sequences('unused',self.tokenizer,'train',16,offset=16)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(both,torch.cat((source,donor))))
        self.assertEqual(rows,srows+drows)
        self.assertFalse(set(r['row'] for r in srows)&set(r['row'] for r in drows))
        self.assertNotEqual([r['row'] for r in rows],sorted(r['row'] for r in rows))
        self.assertEqual(both.shape,(32,128))
        self.assertTrue(all(r['sampling']==SAMPLING_POLICY for r in rows))

    @patch('ilslens.data.load_from_disk')
    def test_short_tokens_skipped_and_insufficient_data_rejected(self,load):
        load.return_value=self.bank()
        with self.assertRaises(ValueError):
            wikitext_sequences('unused',lambda *a,**k:SimpleNamespace(input_ids=[1]*127),'train',1)
        for count,offset,length in [(0,0,128),(1,-1,128),(1,0,0)]:
            with self.assertRaises(ValueError):
                wikitext_sequences('unused',self.tokenizer,'train',count,offset,length)
