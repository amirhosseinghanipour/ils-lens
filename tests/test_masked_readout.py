import unittest
import torch
from ilslens.calibration import calibrated_readout


class MaskedReadoutTests(unittest.TestCase):
    def test_topk_never_resurrects_masked_rows_and_empty_is_unknown(self):
        scores=torch.tensor([9.,8.,7.,float('-inf')])
        calibration=dict(records=[dict(mean=0.,std=1.,threshold=10.,tail=None) for _ in range(4)],
                         dead_mask=torch.tensor([True,True,False,False]))
        result=calibrated_readout(scores,calibration)
        self.assertEqual(result['raw_top10_ids'],[0,1,2])
        self.assertEqual(result['z_top10_ids'],[2])
        self.assertAlmostEqual(result['raw_dead_top10_fraction'],2/3,places=6)
        self.assertEqual(result['z_dead_top10_fraction'],0.)
        calibration['dead_mask'].fill_(True)
        empty=calibrated_readout(scores,calibration)
        self.assertEqual(empty['z_top10_ids'],[])
        self.assertIsNone(empty['z_dead_top10_fraction'])
        self.assertEqual(empty['accepted_ids'],[])
