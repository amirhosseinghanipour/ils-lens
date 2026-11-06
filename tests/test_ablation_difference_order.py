import unittest
import torch
from test_core import tiny_model
from test_runners import script
from ilslens.suffix import QwenSuffix
from ilslens.sampling import horizon_weights

class AblationDifferenceOrderTests(unittest.TestCase):
    def test_large_common_offset_does_not_add_horizon_cancellation(self):
        torch.set_num_threads(2);module=script('collect_ablations')
        cache=QwenSuffix(tiny_model(),torch.randint(0,32,(1,128)))
        cache.states[3].fill_(1000.)
        patch=torch.full((16,),.01);gammas=(0.,.5,.9,1.);position=16
        clean=cache.states[3][0,position];plus=clean+patch;minus=clean-patch
        for kind in ('no_antithetic','gaussian','isotropic'):
            physical=plus-clean if kind=='no_antithetic' else (plus-minus)/2
            expected=torch.stack([physical*horizon_weights(position,128,g)[0] for g in gammas])
            actual=module.ablation_response(cache,3,position,patch,gammas,kind)
            with self.subTest(kind=kind):torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-8)
