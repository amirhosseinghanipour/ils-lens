import unittest
import torch
from test_core import tiny_model
from ilslens.baselines import lrp_rules

class LRPGradientTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(801)
    def test_rmsnorm_vjp_detaches_only_denominator(self):
        model=tiny_model();norm=model.model.layers[0].input_layernorm
        with torch.no_grad():norm.weight.copy_(torch.linspace(.5,1.5,16))
        x=torch.randn(3,16,requires_grad=True);v=torch.randn_like(x);original=norm.forward
        expected=v*norm.weight/(x.detach().square().mean(-1,keepdim=True)+norm.variance_epsilon).sqrt()
        with lrp_rules(model):
            output=norm(x);actual=torch.autograd.grad((output*v).sum(),x)[0]
        torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6)
        self.assertEqual(norm.forward,original)
        conventional=torch.autograd.grad((norm(x)*v).sum(),x)[0]
        self.assertGreater(float((conventional-actual).norm()),.01)
    def test_mlp_vjp_matches_identity_and_half_rules_at_zero_gate(self):
        model=tiny_model();mlp=model.model.layers[0].mlp
        with torch.no_grad():mlp.gate_proj.weight.copy_(torch.eye(16).repeat(2,1))
        x=torch.randn(3,16);x[:,0]=0.;x.requires_grad_(True);v=torch.randn(3,16)
        gate=mlp.gate_proj(x.detach());up=mlp.up_proj(x.detach());activation=gate*gate.sigmoid()
        self.assertTrue((gate[:,0]==0).all());self.assertGreater(float(up[:,0].abs().sum()),0.)
        upstream=v@mlp.down_proj.weight
        expected=.5*((upstream*gate.sigmoid()*up)@mlp.gate_proj.weight+(upstream*activation)@mlp.up_proj.weight)
        clean=mlp(x).detach();original=mlp.forward
        with lrp_rules(model):
            output=mlp(x);actual=torch.autograd.grad((output*v).sum(),x)[0]
        torch.testing.assert_close(output,clean,atol=2e-7,rtol=2e-6)
        torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6)
        self.assertEqual(mlp.forward,original)
        self.assertTrue(torch.isfinite(actual).all())
        with self.assertRaisesRegex(RuntimeError,'test restoration'):
            with lrp_rules(model):raise RuntimeError('test restoration')
        self.assertEqual(mlp.forward,original)
