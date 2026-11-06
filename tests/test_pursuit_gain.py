import unittest
from unittest.mock import patch
import torch
from ilslens.readout import sparse_pursuit

class PursuitGainTests(unittest.TestCase):
    def test_stop_uses_realized_constrained_gain_not_candidate_projection(self):
        generator=torch.Generator().manual_seed(14)
        target=torch.randn(16,generator=generator);dictionary=torch.randn(40,16,generator=generator)
        dictionary/=dictionary.norm(dim=1,keepdim=True)
        with patch('ilslens.readout.random_dictionary_gains',return_value=torch.zeros(100)):
            c17,r17=sparse_pursuit(target,dictionary,k=17)
            c18,r18=sparse_pursuit(target,dictionary,k=18)
        candidates=(dictionary@r17).clamp_min(0).square().masked_fill(c17>0,0)
        candidate=float(candidates.max());realized=float(r17.square().sum()-r18.square().sum())
        self.assertGreater(candidate,.04);self.assertLess(realized,.04);self.assertGreater(realized,0.)
        nulls=[torch.zeros(100) for _ in range(17)]+[torch.full((100,),.04)]
        with patch('ilslens.readout.random_dictionary_gains',side_effect=nulls):
            actual_c,actual_r=sparse_pursuit(target,dictionary,k=18)
        torch.testing.assert_close(actual_c,c17,rtol=0,atol=0)
        torch.testing.assert_close(actual_r,r17,rtol=0,atol=0)
    def test_zero_dictionary_produces_empty_decomposition(self):
        target=torch.ones(4);coefficients,residual=sparse_pursuit(target,torch.zeros(8,4))
        self.assertEqual(int(torch.count_nonzero(coefficients)),0)
        torch.testing.assert_close(residual,target)
