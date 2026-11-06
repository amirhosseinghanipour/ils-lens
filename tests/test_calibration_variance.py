import unittest
import numpy as np
import torch
from ilslens.readout import calibrate_token,zscore,fdr

class CalibrationVarianceTests(unittest.TestCase):
    def test_sparse_tail_preserves_empirical_zscore(self):
        reference=np.zeros(100000,dtype=np.float32);reference[:50]=1.
        result=calibrate_token(reference)
        self.assertEqual(result['std'],float(reference.std()))
        self.assertIsNone(result['tail'])
        self.assertEqual(result['tail_status'],'insufficient_exceedances')
        score=torch.tensor([1.]);z=zscore(score,score.new_tensor([result['mean']]),score.new_tensor([result['std']]))
        self.assertTrue(torch.isfinite(z).all());self.assertGreater(float(z),40.)
        accepted,p=fdr(score.numpy(),z.numpy(),[result])
        self.assertEqual(len(accepted),0);self.assertEqual(p.tolist(),[1.])

    def test_discrete_threshold_can_have_no_exceedances_but_nonzero_variance(self):
        result=calibrate_token(np.repeat([0.,1.],50000))
        self.assertEqual(result['std'],.5);self.assertIsNone(result['tail'])
        self.assertEqual(result['threshold'],1.)

    def test_constant_and_nonfinite_references(self):
        result=calibrate_token(np.ones(100000))
        self.assertEqual(result['std'],0.);self.assertEqual(result['tail_status'],'constant_reference')
        for x in (np.full(100000,np.nan),np.full(100000,np.inf),np.ones((100000,1))):
            with self.assertRaises(ValueError):calibrate_token(x)
