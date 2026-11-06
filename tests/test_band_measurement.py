import unittest
import torch
from test_core import tiny_model
from ilslens.eval.causal import gate_band,prompt_metric

class BandMeasurementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(617)
        self.model=tiny_model();self.ids=torch.randint(0,32,(1,24))
    def call(self,transform,score=lambda x:x[0],covariance=None):
        return gate_band(self.model,self.ids,{1:transform},[17,20],
            {1:torch.eye(16) if covariance is None else covariance},score)
    def test_identity_is_undefined_not_a_failed_direction_claim(self):
        result=self.call(lambda h:h)
        self.assertIsNone(result['passes']);self.assertEqual(result['reason'],'zero_displacement')
        self.assertEqual(result['effect'],0.);self.assertEqual(result['random_effects'],[])
    def test_nonfinite_scores_states_covariance_raise(self):
        for transform,score,covariance in [(lambda h:h+1,lambda x:float('nan'),None),
            (lambda h:h*float('nan'),lambda x:x[0],None),
            (lambda h:h+1,lambda x:x[0],torch.full((16,16),float('nan')))]:
            with self.assertRaises(FloatingPointError):self.call(transform,score,covariance)
        # Exceptions must remove intervention hooks.
        self.assertTrue(torch.isfinite(self.model(self.ids).logits).all())
    def test_degenerate_covariance_and_summary_keep_unknowns(self):
        result=self.call(lambda h:h+1,covariance=torch.zeros(16,16))
        self.assertIsNone(result['passes']);self.assertEqual(result['reason'],'degenerate_covariance')
        rows=[dict(item_id='a',passes=None),dict(item_id='a',passes=True),dict(item_id='b',passes=None)]
        summary=prompt_metric(rows,'passes')
        self.assertEqual(summary['mean'],1.);self.assertEqual(summary['undefined_cases'],2)
        self.assertEqual(summary['undefined_items'],1)
        unknown=prompt_metric([rows[0]],'passes')
        self.assertIsNone(unknown['mean']);self.assertEqual(unknown['ci'],[None,None])

    def test_controls_match_every_realized_site_norm(self):
        from unittest.mock import patch
        from ilslens.eval.causal import intervene_band
        realized=[]
        def instrument(model,transforms,positions):
            measurements={};realized.append(measurements)
            def wrap(layer,transform):
                def call(h):
                    out=transform(h);measurements[layer]=(out-h).detach().clone();return out
                return call
            return intervene_band(model,{l:wrap(l,t) for l,t in transforms.items()},positions)
        scales=torch.tensor([.1,.3])[None,:,None]
        covariance=torch.diag(torch.tensor([1.]+[0.]*15))
        with patch('ilslens.eval.causal.intervene_band',side_effect=instrument):
            result=gate_band(self.model,self.ids,{0:lambda h:h+scales,2:lambda h:h-scales},
                [17,20],{0:covariance,2:covariance},lambda x:x[0],seed=83)
        self.assertEqual(result['status'],'measured');self.assertEqual(len(realized),101)
        for control in realized[1:]:
            for layer in (0,2):
                torch.testing.assert_close(control[layer].norm(dim=-1),realized[0][layer].norm(dim=-1),atol=2e-7,rtol=2e-6)
                torch.testing.assert_close(control[layer][...,1:],torch.zeros_like(control[layer][...,1:]),atol=0,rtol=0)
