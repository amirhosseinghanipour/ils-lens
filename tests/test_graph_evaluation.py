import unittest
import torch
from test_core import tiny_model
from ilslens.transcoders.train import SkipTranscoder
from ilslens.graphs.qwen import QwenGraph
from ilslens.graphs.evaluate import evaluate_target
from ilslens.eval.causal import gate_band

class GraphEvaluationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(91)
    def test_multiple_feature_error_embedding_edges_and_original_norm_only(self):
        model=tiny_model();ids=torch.randint(0,32,(1,24))
        transcoders=[SkipTranscoder(16).eval().requires_grad_(False) for _ in range(4)]
        graph=QwenGraph(model,ids,transcoders)
        for target in [dict(kind='lens',layer=2,position=20,vector=torch.randn(16)),dict(kind='logit',position=23,token=3,contrast_token=5)]:
            result=evaluate_target(graph,target,edges_per_kind=2)
            self.assertTrue(result['all_edges_pass'],result['edge_checks'])
            self.assertEqual(len(result['edge_checks']),6)
            self.assertTrue(all('norm_only_effect' in edge for edge in result['edge_checks']))
            for edge in result['edge_checks']:
                live=edge['live_replacement']
                self.assertIn('support_changes',live)
                self.assertEqual(live['passes'],live['relative_error']<1e-3)
                self.assertEqual(live['direct_edge_prediction'],edge['predicted'])
        torch.testing.assert_close(model(ids).logits,graph.clean_logits)
    def test_band_random_gate_matches_site_norms(self):
        model=tiny_model();ids=torch.randint(0,32,(1,24));vector=torch.randn(16)*.01
        result=gate_band(model,ids,{1:lambda h:h+vector},[17,20],{1:torch.eye(16)},lambda logits:logits.softmax(-1)[3],seed=92)
        self.assertEqual(len(result['random_effects']),100)
        torch.testing.assert_close(torch.tensor(result['displacement_norms'][1]),vector.norm().expand(1,2))
        self.assertEqual(result['passes'],result['effect']>result['threshold'])
