import unittest
from unittest.mock import patch
import torch
from ilslens.readout import sparse_pursuit

class PursuitGainTests(unittest.TestCase):
    def test_stop_uses_realized_constrained_gain_not_candidate_projection(self):
        generator=torch.Generator().manual_seed(18)
        target=torch.randn(16,generator=generator);dictionary=torch.randn(40,16,generator=generator)
        dictionary/=dictionary.norm(dim=1,keepdim=True)
        with patch('ilslens.readout.random_dictionary_gains',return_value=torch.zeros(100)):
            c5,r5=sparse_pursuit(target,dictionary,k=5)
            c6,r6=sparse_pursuit(target,dictionary,k=6)
        candidates=(dictionary@r5).clamp_min(0).square().masked_fill(c5>0,0)
        candidate=float(candidates.max());realized=float(r5.square().sum()-r6.square().sum())
        self.assertGreater(candidate,.9);self.assertLess(realized,.9);self.assertGreater(realized,0.)
        nulls=[torch.zeros(100) for _ in range(5)]+[torch.full((100,),.9)]
        with patch('ilslens.readout.random_dictionary_gains',side_effect=nulls):
            actual_c,actual_r=sparse_pursuit(target,dictionary,k=6)
        torch.testing.assert_close(actual_c,c5,rtol=0,atol=0)
        torch.testing.assert_close(actual_r,r5,rtol=0,atol=0)
    def test_zero_dictionary_produces_empty_decomposition(self):
        target=torch.ones(4);coefficients,residual=sparse_pursuit(target,torch.zeros(8,4))
        self.assertEqual(int(torch.count_nonzero(coefficients)),0)
        torch.testing.assert_close(residual,target)
