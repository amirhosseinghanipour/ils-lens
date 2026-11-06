import unittest
import torch
from test_core import tiny_model
from ilslens.eval.faithfulness import horizon_log_probability_changes
from ilslens.sampling import horizon_weights

class HorizonSharingTests(unittest.TestCase):
    def test_shared_readout_matches_independent_horizon_definitions(self):
        torch.set_num_threads(2);torch.manual_seed(24);model=tiny_model()
        plus=torch.randn(11,16);minus=torch.randn(11,16)
        weights=torch.stack([horizon_weights(16,28,gamma) for gamma in (0.,.5,.9,1.)])
        calls=[];handle=model.lm_head.register_forward_hook(lambda *args:calls.append(1))
        try:actual=horizon_log_probability_changes(model,plus,minus,weights,position_batch=3)
        finally:handle.remove()
        self.assertEqual(len(calls),8)  # two branches, four position chunks, any number of horizons
        with torch.no_grad():
            per_position=(model.lm_head(model.model.norm(plus)).log_softmax(-1)-model.lm_head(model.model.norm(minus)).log_softmax(-1))/2
            expected=torch.stack([sum(weight[t]*per_position[t] for t in range(11)) for weight in weights])
        torch.testing.assert_close(actual,expected,atol=3e-7,rtol=1e-5)
