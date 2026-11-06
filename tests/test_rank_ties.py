import unittest
import torch
from ilslens.readout import target_ranks,best_target_rank
from ilslens.calibration import finite_topk
from ilslens.contrast import rank_of


class RankTieTests(unittest.TestCase):
    def test_zero_lens_cannot_assign_every_token_rank_one(self):
        scores=torch.zeros(16)
        ranks=torch.stack([target_ranks(scores,[i]) for i in range(16)])
        torch.testing.assert_close(ranks,torch.arange(1,17,dtype=torch.float32))
        self.assertEqual(rank_of(scores,[9,3]),4)
        self.assertEqual(finite_topk(scores,10).tolist(),list(range(10)))

    def test_ranks_agree_with_stable_topk_for_aliases_and_masks(self):
        scores=torch.tensor([[2.,3.,3.,1.,-torch.inf,0.],[1.,0.,2.,2.,-torch.inf,1.]])
        for row in scores:
            order=finite_topk(row,6).tolist()
            for token in range(6):
                rank=target_ranks(row,[token])
                if token in order:self.assertEqual(int(rank),order.index(token)+1)
                else:self.assertTrue(torch.isnan(rank))
        self.assertEqual(best_target_rank(scores,[0,3]),2.)
        self.assertEqual(float(target_ranks(torch.tensor([float('nan'),1.,float('inf'),0.]),[0,1])),1.)
        with self.assertRaises(ValueError):target_ranks(scores,[-1])
