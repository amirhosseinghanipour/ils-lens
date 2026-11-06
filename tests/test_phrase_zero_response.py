import unittest
import torch
from test_core import tiny_model
from ilslens.templatelens import antithetic_phrase_effect,fit_phrase_rows
from ilslens.fit import ridge


class PhraseZeroResponseTests(unittest.TestCase):
    def test_final_block_cannot_affect_later_probe_phrase(self):
        torch.set_num_threads(2);model=tiny_model();prompt=list(range(24))
        for position in (16,20,23):
            effect=antithetic_phrase_effect(model,prompt,[7,8],[9,10],3,position,torch.randn(16)*3)
            self.assertEqual(float(effect),0.)

    def test_zero_response_has_no_invented_penalty_or_r2(self):
        torch.manual_seed(4);x=torch.randn(40,4);groups=torch.arange(20).repeat_interleave(2)
        y=torch.stack((torch.zeros(40),x[:,0]),dim=1)
        rows,selection,cv=fit_phrase_rows(x,y,groups,[0.,.01,1.])
        torch.testing.assert_close(rows[0],torch.zeros(4),atol=0,rtol=0)
        self.assertEqual(selection[0]['status'],'exact_zero_response')
        self.assertIsNone(selection[0]['lam']);self.assertIsNone(selection[0]['r2']);self.assertEqual(cv[0],[])
        self.assertEqual(selection[1]['status'],'cv_selected')
        for penalty in (0.,.01,1.):
            torch.testing.assert_close(ridge(x,y[:,:1],penalty,torch.zeros(1,4))[0],rows[0],atol=0,rtol=0)
