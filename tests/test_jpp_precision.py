import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from ilslens.jpp import save_fp32_experts,require_fp32_experts


class JppPrecisionTests(unittest.TestCase):
    def fixture(self):
        value=torch.tensor([[[1.+2**-20]]],dtype=torch.float32)
        return SimpleNamespace(experts_L_dict_EFN={0:value},pooled_jacobian_L_dict_FN={0:value[0]},
            position_counts_L_dict_E={0:torch.tensor([100])},weight_sums_L_dict_E={0:torch.tensor([100.])},
            fallback_L_dict_Bool_E={0:torch.tensor([False])},min_kept_positions=50,
            config=SimpleNamespace(to_dict=lambda:dict(fixture=True)))

    def test_save_preserves_bits_that_fp16_would_lose(self):
        experts=self.fixture();value=experts.experts_L_dict_EFN[0]
        self.assertFalse(torch.equal(value,value.half().float()))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'experts.pt';save_fp32_experts(experts,path)
            saved=torch.load(path,weights_only=True)
            for key,expected in [('experts',value),('pooled',value[0])]:
                self.assertEqual(saved[key][0].dtype,torch.float32)
                self.assertTrue(torch.equal(saved[key][0],expected))
            self.assertEqual(saved['storage_protocol'],'fp32-experts-v1')

    def test_rounded_and_nonfinite_experts_are_rejected(self):
        for value in (torch.ones(1,1,1,dtype=torch.float16),torch.full((1,1,1),float('nan'))):
            experts=self.fixture();experts.experts_L_dict_EFN[0]=value
            with self.assertRaises(ValueError):require_fp32_experts(experts)
