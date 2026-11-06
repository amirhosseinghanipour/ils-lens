import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from test_core import tiny_model
from ilslens.eval.readout_tasks import ReadoutItem,gate_item,recall_report,contains_concept,dev_split
from ilslens.eval.causal import swap_coordinates,ablate_direction,intervene_band
from ilslens.templatelens import continuation_log_probability,antithetic_phrase_effect
from ilslens.readout import fdr
from ilslens.runtime import resolve_device

class EvaluationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(99)
    def test_capability_and_leakage_are_both_required(self):
        item=ReadoutItem('a','multihop','The country shaped like a boot uses','same',['euro'],['Italy'],'test')
        clean=gate_item(item,['euro']*8+['dollar']*2,['euro']*10)
        self.assertTrue(clean['retained'])
        leaky=gate_item(item,['euro']*10,['Italy uses the euro']+['euro']*9)
        self.assertFalse(leaky['retained'])
        self.assertFalse(contains_concept('earth','art'))
        self.assertTrue(contains_concept('Italy, of course','Italy'))
        self.assertEqual(dev_split('fixed'),dev_split('fixed'))
    def test_macro_recall_preserves_unscoreable_items(self):
        ranks=np.array([[[1,np.nan],[20,np.nan]],[[5,20],[2,3]],[[np.nan,np.nan],[np.nan,np.nan]]])
        rows=recall_report(ranks,['a','b','c'])
        row=next(r for r in rows if r['layer']=='any' and r['k']==1)
        self.assertEqual(row['recall'],.5);self.assertEqual(row['n'],2)
        self.assertEqual(row['unscoreable_items'],1)
    def test_swap_is_involution_and_ablation_is_projection(self):
        h=torch.randn(6,8);v=torch.randn(8,2)
        swapped=swap_coordinates(h,v)
        torch.testing.assert_close(swap_coordinates(swapped,v),h,atol=2e-6,rtol=1e-5)
        residual=h-(h@torch.linalg.pinv(v).T)@v.T
        torch.testing.assert_close(swapped-(swapped@torch.linalg.pinv(v).T)@v.T,residual,atol=2e-6,rtol=1e-5)
        ablated=ablate_direction(h,v[:,0])
        torch.testing.assert_close(ablated@v[:,0],torch.zeros(6),atol=2e-6,rtol=0)
    def test_band_hooks_restore_after_exception(self):
        model=tiny_model();ids=torch.randint(0,32,(1,24));clean=model(ids).logits
        with self.assertRaises(RuntimeError):
            with intervene_band(model,{0:lambda x:x+1},[18]):
                raise RuntimeError('synthetic exception')
        torch.testing.assert_close(model(ids).logits,clean)
    def test_phrase_teacher_forcing_matches_sequential_probabilities(self):
        model=tiny_model();prefix=[1,2,3];phrase=[4,5,6]
        expected=0.
        for i,token in enumerate(phrase):
            expected+=model(torch.tensor([prefix+phrase[:i]])).logits[0,-1].log_softmax(0)[token]
        torch.testing.assert_close(continuation_log_probability(model,prefix,phrase),expected)
        delta=torch.randn(16)*.01
        plus=antithetic_phrase_effect(model,prefix,[7],phrase,0,1,delta)
        minus=antithetic_phrase_effect(model,prefix,[7],phrase,0,1,-delta)
        torch.testing.assert_close(plus,-minus)
        self.assertEqual(antithetic_phrase_effect(model,prefix,[7],phrase,0,1,torch.zeros(16)),0)
    def test_full_vocabulary_by_not_candidate_only(self):
        v=1000;scores=np.zeros(v);scores[0]=10.;z=np.arange(v,dtype=float);z[0]=2000.
        calibration=[dict(threshold=0,tail=None) for _ in range(v)]
        # For exponential tails p=.05*exp(-10)=2.27e-6: would pass
        # candidate-only Bonferroni but not full-vocabulary BY rank one.
        calibration[0]=dict(threshold=0,tail=(0,1),exceedance=.05)
        accepted,p=fdr(scores,z,calibration,q=.01)
        self.assertEqual(len(accepted),0);self.assertTrue(np.all(p[1:]==1))
    @patch('ilslens.runtime.subprocess.run')
    @patch('ilslens.runtime.torch.cuda.is_available')
    def test_busy_gpu_rejected_before_cuda_initialization(self,available,run):
        run.return_value=SimpleNamespace(stdout='4096, 100\n')
        with self.assertRaisesRegex(RuntimeError,'GPU busy'):resolve_device('cuda')
        available.assert_not_called()
