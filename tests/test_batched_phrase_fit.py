import unittest
from unittest.mock import patch
import torch
from ilslens.fit import select,ridge
from ilslens.templatelens import fit_phrase_rows


class BatchedPhraseFitTests(unittest.TestCase):
    def test_same_folds_penalties_and_maps_as_scalar_reference(self):
        torch.set_num_threads(2);torch.manual_seed(109)
        x=torch.randn(100,8);y=x@torch.randn(8,7)+torch.randn(100,7)*.3
        y[:,3]=0;groups=torch.arange(20).repeat_interleave(5);penalties=[0.,.001,.1,1.]
        with patch('ilslens.templatelens.ridge',wraps=ridge) as calls:
            matrices,winners,cv=fit_phrase_rows(x,y,groups,penalties,seed=9)
        self.assertLessEqual(calls.call_count,len(penalties)*6)
        for column in range(7):
            if column==3:
                self.assertEqual(winners[column]['status'],'exact_zero_response');continue
            expected,winner,reference=select(x,y[:,column:column+1],groups,{'I':torch.zeros(1,8)},penalties,estimators=('ridge',),seed=9)
            self.assertEqual(winners[column]['lam'],winner['lam'])
            torch.testing.assert_close(matrices[column],expected[0],atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(torch.tensor([r['r2'] for r in cv[column]]),torch.tensor([r['r2'] for r in reference]),atol=1e-6,rtol=1e-6)

    def test_nonzero_constant_target_does_not_get_an_arbitrary_fit(self):
        with self.assertRaisesRegex(ValueError,'nonzero constant'):
            fit_phrase_rows(torch.randn(20,4),torch.ones(20,2),torch.arange(20),[0.,1.])
