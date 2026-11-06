import unittest
import torch
from test_runners import script

class PrimaryResourceTests(unittest.TestCase):
    def test_tensor_layout_and_batched_timing_are_counted_once(self):
        module=script('audit_primary_resources')
        config=dict(samples_per_layer_epsilon=16,heldout_samples_per_layer_epsilon=8,epsilons=[.1,.2],gammas=[0.,.5,1.],source_prompts=4,donor_prompts=4,baseline_corpus='source_and_donor')
        benchmark=dict(epsilon_batch=2,measurements=[dict(layer=0,seconds_per_six_epsilon_sample=3.),dict(layer=2,seconds_per_six_epsilon_sample=1.)])
        result=module.estimate(config,benchmark,layers=3,dimension=4)
        tensors=[torch.empty(24,3,2,4),torch.empty(24,3,2,3,4),torch.empty(24,3,4,dtype=torch.int64),torch.empty(2,8,3,4,4),torch.empty(3,2,3,4,4)]
        self.assertEqual(result['tensor_bytes_lower_bound'],sum(t.numel()*t.element_size() for t in tensors))
        self.assertEqual(result['suffix_projection_seconds'],24*(3+2+1))
        benchmark['epsilon_batch']=3
        with self.assertRaises(ValueError):module.estimate(config,benchmark,layers=3,dimension=4)
