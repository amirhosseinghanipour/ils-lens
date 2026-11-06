import json,unittest
from pathlib import Path
import numpy as np
import torch
from ilslens.eval.faithfulness import bootstrap_r2,bootstrap_prompt_mean,token_spearman
from ilslens.eval.readout_tasks import load_simplified_items,gate_item
from ilslens.fit import select
from ilslens.eval.faithfulness import horizon_log_probability_change
from test_core import tiny_model

class DegenerateMeasurementTests(unittest.TestCase):
    def test_token_changes_normalize_before_horizon_averaging(self):
        model=tiny_model();torch.manual_seed(73)
        plus=torch.randn(5,16)*torch.arange(1,6)[:,None];minus=torch.randn(5,16)
        weights=torch.tensor([.1,.2,.3,.25,.15])
        lp=lambda h:model.lm_head(model.model.norm(h)).log_softmax(-1)
        expected=((lp(plus)-lp(minus))*weights[:,None]).sum(0)/2
        actual=horizon_log_probability_change(model,plus,minus,weights,position_batch=2)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        wrong=(lp((plus*weights[:,None]).sum(0))-lp((minus*weights[:,None]).sum(0)))/2
        self.assertGreater(float((actual-wrong).norm()),.01)
    def test_zero_effect_is_undefined_not_a_perfect_fit_or_serialization_failure(self):
        value=bootstrap_r2(torch.zeros(12,4),torch.zeros(12,4),torch.arange(12),repeats=20)
        self.assertIsNone(value['r2']);self.assertEqual(value['ci'],[None,None])
        self.assertEqual(value['undefined_bootstrap_resamples'],20);json.dumps(value,allow_nan=False)
        with self.assertRaisesRegex(ValueError,'undefined'):
            select(torch.randn(12,4),torch.zeros(12,4),torch.arange(12),{'I':torch.eye(4)},[0.],estimators=('ridge',))
    def test_undefined_spearman_is_counted_by_prompt(self):
        self.assertTrue(np.isnan(token_spearman(torch.zeros(10),torch.zeros(10))))
        result=bootstrap_prompt_mean([[np.nan,np.nan],[.5,np.nan],[1.,1.]])
        self.assertEqual(result['mean'],.75);self.assertEqual(result['valid_prompts'],2)
        self.assertEqual(result['undefined_measurements'],3);self.assertEqual(result['undefined_prompts'],1)
        absent=bootstrap_prompt_mean(np.full((4,2),np.nan));self.assertIsNone(absent['mean']);json.dumps(absent,allow_nan=False)
    def test_simplified_bank_is_separate_and_keeps_strict_poetry_leakage(self):
        path=Path(__file__).resolve().parents[1]/'configs/readout-simple.json'
        items=load_simplified_items(path,['multihop','multilingual','typo','association','poetry'],seed=201)
        self.assertEqual(len(items),36);self.assertTrue(all(item.task.endswith(':simplified') for item in items))
        self.assertEqual(len({item.id for item in items}),len(items))
        for item in items:
            if item.task=='poetry:simplified':
                result=gate_item(item,[item.answers[0]]*10,[item.answers[0]]*10)
                self.assertEqual(result['correct'],10);self.assertFalse(result['retained'])
