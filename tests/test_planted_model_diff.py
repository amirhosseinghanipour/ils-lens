import copy,unittest
import torch
from test_core import tiny_model
from ilslens.lora import AdapterSet
from ilslens.hooks import capture_blocks
from ilslens.collect import antithetic
from ilslens.eval.planted import model_diff_faithfulness

class PlantedModelDiffTests(unittest.TestCase):
    def test_effect_uses_independent_base_model_on_matched_short_context(self):
        torch.set_num_threads(2);torch.manual_seed(431)
        model=tiny_model();base=copy.deepcopy(model);adapters=AdapterSet(model,rank=2,alpha=4)
        with torch.no_grad():
            for module in adapters.modules.values():module.b.normal_(std=.15)
        ids=torch.randint(0,32,(1,12));position=4;layer=1
        with torch.no_grad(),capture_blocks(base,[layer]) as b:base.model(ids,use_cache=False)
        with torch.no_grad(),capture_blocks(model,[layer]) as t:model.model(ids,use_cache=False)
        delta=t[layer][0,position]-b[layer][0,position]
        expected=antithetic(base,ids,layer,position,delta,gammas=(.9,),skip=0)[0,0]
        actual=model_diff_faithfulness(model,adapters,[ids[0]],layer,torch.eye(16),[position],gamma=.9,skip=0)
        self.assertGreater(delta.norm().item(),0.)
        torch.testing.assert_close(actual['delta'][0],delta)
        torch.testing.assert_close(actual['effect'][0],expected)
        self.assertTrue(all(m.enabled for m in adapters.modules.values()))
        adapters.remove()
    def test_context_count_mismatch_is_not_silently_truncated(self):
        for sequences,positions in [([],[]),([torch.zeros(12,dtype=torch.long)],[])]:
            with self.assertRaises(ValueError):
                model_diff_faithfulness(None,None,sequences,0,torch.eye(4),positions)
