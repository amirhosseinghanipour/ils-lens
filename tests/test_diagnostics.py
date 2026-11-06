import json,unittest
import numpy as np
import torch
from test_core import tiny_model
from ilslens.eval.numerical import jacobian_rows,relative_jvp_error
from ilslens.eval.diagnostics import balanced_prefix
from ilslens.templatelens import fit_phrase_rows
from ilslens.concept import projection
from ilslens.readout import random_dictionary_gains,sparse_pursuit
from ilslens.suffix import QwenSuffix

class DiagnosticsTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(77)
    def test_batched_jacobian_and_jvp_on_nonlinear_map(self):
        w=torch.randn(7,5);x=torch.randn(5)*.1;f=lambda x:(w@x).tanh()
        expected=torch.autograd.functional.jacobian(f,x)
        torch.testing.assert_close(jacobian_rows(f,x,batch_size=3),expected)
        self.assertLess(relative_jvp_error(f,x,torch.randn(5)*.001),.001)
    def test_learning_prefix_is_nested_and_prompt_balanced(self):
        groups=torch.arange(16).repeat_interleave(448)
        previous=torch.tensor([],dtype=torch.long)
        for n in (896,1792,3584,7168):
            index=balanced_prefix(groups,n)
            torch.testing.assert_close(index[:len(previous)],previous)
            self.assertEqual(len(index.unique()),n);self.assertEqual(groups[index].unique().numel(),16)
            self.assertEqual(torch.bincount(groups[index]).unique().numel(),1);previous=index
    def test_phrase_rows_use_independent_cv(self):
        x=torch.randn(100,4);a=torch.randn(2,4);y=x@a.T;groups=torch.arange(10).repeat_interleave(10)
        fitted,winners,cv=fit_phrase_rows(x,y,groups,[0.,.1])
        torch.testing.assert_close(fitted,a,atol=2e-5,rtol=2e-5)
        self.assertEqual(len(winners),2);self.assertTrue(all(w['lam']==0 for w in winners))
    def test_concept_projection_removes_shared_horizon(self):
        short=torch.diag(torch.tensor([1.,1.,0.,0.]));long=torch.eye(4)
        b=projection(short,long,torch.eye(4),1e-6)
        torch.testing.assert_close(long-b@short,torch.diag(torch.tensor([0.,0.,1.,1.])),atol=2e-6,rtol=1e-5)
    def test_exact_null_sampler_agrees_with_explicit_random_dictionaries(self):
        target=torch.randn(16);generator=torch.Generator().manual_seed(17)
        exact=random_dictionary_gains(target,8,20000,generator)
        dictionaries=torch.randn(20000,8,16);dictionaries/=dictionaries.norm(dim=-1,keepdim=True)
        explicit=(dictionaries@target).clamp_min(0).square().max(1).values
        self.assertLess(abs(float(exact.quantile(.95)-explicit.quantile(.95)))/float(explicit.quantile(.95)),.04)
        dictionary=torch.cat([target[None],torch.randn(7,16)])
        coefficients,residual=sparse_pursuit(target,dictionary)
        self.assertTrue((coefficients>=0).all());self.assertLess(float(residual.norm()),1e-5)
    def test_suffix_branch_mean_matches_effects(self):
        model=tiny_model();cache=QwenSuffix(model,torch.randint(0,32,(1,24)));delta=torch.randn(3,16)*.01
        plus,minus=cache.effects(1,19,delta,return_branches=True)
        effects=cache.effects(1,19,delta)
        torch.testing.assert_close((plus-minus)/2,effects,atol=2e-7,rtol=1e-4)
