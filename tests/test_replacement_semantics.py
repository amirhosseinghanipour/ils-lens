import unittest
import torch
from test_core import tiny_model
from ilslens.graphs.qwen import QwenGraph
from ilslens.graphs.intervention import replacement_effect
from ilslens.transcoders.train import SkipTranscoder


class ReplacementSemanticsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model=tiny_model()
        self.ids=torch.randint(0,32,(1,24))
        self.transcoders=[SkipTranscoder(16).eval().requires_grad_(False) for _ in range(4)]
        self.graph=QwenGraph(self.model,self.ids,self.transcoders)

    def test_clean_replacement_and_live_upstream_path(self):
        with self.graph.replacement():
            logits=self.graph.forward()
            torch.testing.assert_close(logits,self.graph.clean_logits,atol=2e-6,rtol=1e-5)
            feature=self.graph.nodes['feature:2']
            grad=torch.autograd.grad(feature.sum(),self.graph.nodes['embedding'])[0]
            self.assertGreater(float(grad.norm()),0)
        torch.testing.assert_close(self.model(self.ids).logits,self.graph.clean_logits)

    def test_final_block_edge_is_exact_without_extra_feature_freeze(self):
        delta=torch.zeros_like(self.graph.outputs[3]);delta[0,20]=torch.randn(16)*.1
        target=dict(kind='lens',layer=3,position=20,vector=torch.randn(16))
        result=replacement_effect(self.graph,target,3,delta)
        self.assertTrue(result['passes'],result)
        self.assertEqual(sum(result['support_changes'].values()),0)

    def test_support_switch_is_measured_not_silently_frozen(self):
        delta=torch.zeros_like(self.graph.outputs[0]);delta[0,20]=torch.randn(16)*10
        target=dict(kind='lens',layer=3,position=20,vector=torch.randn(16))
        result=replacement_effect(self.graph,target,0,delta)
        self.assertGreater(sum(result['support_changes'].values()),0)
        self.assertFalse(result['passes'],result)
        torch.testing.assert_close(self.model(self.ids).logits,self.graph.clean_logits)
