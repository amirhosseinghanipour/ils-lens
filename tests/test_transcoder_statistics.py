import unittest,json
import torch
from ilslens.transcoders.eval import output_statistics,explained_from_statistics,uncertainty,ce_recovered

class TranscoderStatisticsTests(unittest.TestCase):
    def test_centered_moments_preserve_small_variance_and_bootstrap_duplicates(self):
        torch.manual_seed(818)
        y=(10000+torch.randn(19,4)*.01).float();pred=y+.001
        chunks=[output_statistics(pred[:7],y[:7]),output_statistics(pred[7:],y[7:])]
        for rows,indices in [(chunks,torch.arange(19)),([chunks[1],chunks[0],chunks[1]],torch.cat([torch.arange(7,19),torch.arange(7),torch.arange(7,19)]))]:
            target=y[indices].double();expected=1-(pred[indices].double()-target).square().sum()/(target-target.mean(0)).square().sum()
            self.assertAlmostEqual(explained_from_statistics(rows),float(expected),places=10)
    def test_constant_outputs_and_equal_ce_are_explicitly_undefined(self):
        row=output_statistics(torch.zeros(3,4),torch.ones(3,4))
        value=explained_from_statistics([row]);summary=uncertainty(value,[value]*200,'undefined_zero_output_variance')
        self.assertIsNone(summary['value']);self.assertEqual(summary['undefined_bootstrap_resamples'],200)
        json.dumps(summary,allow_nan=False)
        value=ce_recovered(1.,1.,1.);summary=uncertainty(value,[value]*200,'undefined_equal_clean_and_ablated_loss')
        self.assertIsNone(summary['value']);self.assertEqual(summary['ci'],[None,None])
    def test_undefined_bootstrap_draws_are_counted(self):
        summary=uncertainty(-2.,[-3.,-1.]*20+[float('nan')]*5,'undefined')
        self.assertEqual(summary['value'],-2.);self.assertEqual(summary['undefined_bootstrap_resamples'],5)
        self.assertEqual(summary['ci'],[-3.,-1.])
        self.assertEqual(uncertainty(1.,[1.]*19,'undefined')['ci'],[None,None])
