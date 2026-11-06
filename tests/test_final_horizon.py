import unittest
import torch
from test_core import tiny_model
from ilslens.eval.final_horizon import final_horizon_diagnostic
from ilslens.sampling import horizon_weights
from ilslens.suffix import QwenSuffix

class FinalHorizonTests(unittest.TestCase):
    def test_actual_suffix_obeys_position_dependent_linear_map(self):
        torch.set_num_threads(2);model=tiny_model()
        ids=torch.randint(0,32,(1,128));cache=QwenSuffix(model,ids)
        generator=torch.Generator().manual_seed(317)
        delta=torch.randn(7,16,generator=generator)*.1
        for position in (16,64,126):
            actual=cache.effects(3,position,delta)
            expected=torch.stack([horizon_weights(position,128,g)[0]*delta for g in (0.,.5,.9,1.)],dim=1)
            torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-5)

    def test_population_formula_matches_balanced_basis_dataset(self):
        d=3;x=torch.cat((torch.eye(d),-torch.eye(d))).repeat(111,1)
        for row in final_horizon_diagnostic()['rows']:
            gains=torch.stack([horizon_weights(t,128,row['gamma'])[0] for t in range(16,127)])
            y=x*gains.repeat_interleave(2*d)[:,None]
            denominator=y.square().sum()
            for gain,key in [(row['mean_gain'],'optimal_global_isotropic_r2'),(1.,'upstream_identity_isotropic_r2')]:
                measured=(1-(y-x*gain).square().sum()/denominator).item()
                self.assertAlmostEqual(measured,row[key],places=4)
        rows=final_horizon_diagnostic()['rows']
        self.assertEqual(rows[0]['optimal_global_isotropic_r2'],1.)
        self.assertLess(rows[-1]['optimal_global_isotropic_r2'],.2)
