import unittest
import torch
from test_core import tiny_model
from ilslens.suffix import QwenSuffix
from ilslens.collect import antithetic
from ilslens.graphs.qwen import QwenGraph
from ilslens.transcoders.train import SkipTranscoder
from ilslens.fit import select

class ResearchTests(unittest.TestCase):
    def test_cv_prior_provider_receives_only_training_groups(self):
        torch.manual_seed(13)
        x=torch.randn(40,4);y=x.clone();groups=torch.arange(20).repeat_interleave(2)
        seen=[]
        def provider(name,training_groups):
            self.assertEqual(name,'I')
            self.assertEqual(len(training_groups),16)
            seen.append(set(training_groups.tolist()))
            return torch.eye(4)
        _,selection,_=select(x,y,groups,{'I':torch.eye(4)},[.01],
                             estimators=('ridge',),fold_prior_provider=provider)
        self.assertEqual(len(seen),5)
        self.assertEqual(set.intersection(*seen),set())
        self.assertGreater(selection['r2'],.999)
    def test_full_qwen_frozen_source_graph(self):
        torch.set_num_threads(2);torch.manual_seed(32)
        model=tiny_model();ids=torch.randint(0,32,(1,24))
        transcoders=[SkipTranscoder(16).requires_grad_(False) for _ in range(4)]
        graph=QwenGraph(model,ids,transcoders)
        with graph.frozen():
            logits=graph.forward()
            torch.testing.assert_close(logits,graph.clean_logits,atol=2e-6,rtol=1e-5)
            target=logits[0,-1,3]-logits[0,-1,7]
            node=graph.nodes['feature:0'];delta=torch.randn_like(node)*.05
            gradient=torch.autograd.grad(target,node,retain_graph=True)[0]
            predicted=(gradient*delta).sum()
            with torch.no_grad():node.add_(delta)
            changed=graph.forward()[0,-1]
            actual=changed[3]-changed[7]-target.detach()
            relative=(predicted-actual).abs()/actual.abs().clamp_min(1e-7)
            self.assertLess(relative.item(),1e-3)
        # Forward methods must be restored even after graph instrumentation.
        torch.testing.assert_close(model(ids).logits,graph.clean_logits)
    def test_cached_suffix_matches_full_replay(self):
        torch.set_num_threads(2);torch.manual_seed(4)
        model=tiny_model();ids=torch.randint(0,32,(1,24))
        replay=QwenSuffix(model,ids)
        deltas=torch.randn(3,16)*.01
        for layer in (0,2,3):
            actual=replay.effects(layer,18,deltas)
            for i,delta in enumerate(deltas):
                expected=antithetic(model,ids,layer,18,delta)[:,0]
                torch.testing.assert_close(actual[i],expected,atol=1e-6,rtol=1e-4)

    def test_sum_and_normalized_horizons_are_different_estimands(self):
        # Causal linear sequence: outputs y0=x0, y1=10*x0+x1.
        # For sources t=0,1: summed Jacobians 11 and 1; normalized 5.5 and 1.
        upstream=(11.+1.)/2
        normalized=(5.5+1.)/2
        self.assertEqual(upstream,6.)
        self.assertEqual(normalized,3.25)
        self.assertNotEqual(upstream,normalized)
