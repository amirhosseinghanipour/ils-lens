import unittest
import importlib.util
import os
from pathlib import Path
import torch
from ilslens.fit import ridge,fit,geometric_median,select
from ilslens.baselines import lrp_rules
from ilslens.collect import antithetic
from ilslens.hooks import capture_blocks,final_state,patch_site
from ilslens.sampling import horizon_weights,donor
from ilslens.graphs.replacement import FrozenTranscoder
from ilslens.graphs.validation import validate_edge
from ilslens.transcoders.train import SkipTranscoder
from ilslens.transcoders.eval import ce_recovered
from transformers import Qwen2Config,Qwen2ForCausalLM

def tiny_model():
    torch.manual_seed(1)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,
        intermediate_size=32,num_hidden_layers=4,num_attention_heads=2,
        num_key_value_heads=1,max_position_embeddings=128,
        attn_implementation='eager')).float().eval().requires_grad_(False)

class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)
    def test_quadratic_antithetic(self):
        h,d=torch.randn(8),torch.randn(8)
        q=torch.randn(8,8); q=(q+q.T)/2
        f=lambda x:x@q@x
        torch.testing.assert_close((f(h+d)-f(h-d))/2,2*h@q@d)
    def test_ridge_iterative(self):
        x,y=torch.randn(80,6),torch.randn(80,6)
        prior=torch.eye(6); lam=.1
        a=torch.zeros(6,6,requires_grad=True)
        optimizer=torch.optim.LBFGS([a],max_iter=100,tolerance_grad=1e-9,tolerance_change=1e-12)
        def closure():
            optimizer.zero_grad()
            loss=(y-x@a.T).square().sum()/len(x)+lam*(a-prior).square().sum()
            loss.backward(); return loss
        optimizer.step(closure)
        torch.testing.assert_close(a,ridge(x,y,lam,prior),atol=1e-5,rtol=1e-4)
    def test_weiszfeld(self):
        points=torch.tensor([[0.,0.],[2.,0.],[1.,3.]])
        result=geometric_median(points)
        torch.testing.assert_close(result,torch.tensor([1.,3**-.5]),atol=1e-4,rtol=1e-4)
        # Coincidence with a non-optimal point must not cause early termination.
        result=geometric_median(torch.tensor([[0.],[1.],[1.],[1.],[2.],[7.]]))
        self.assertAlmostEqual(result.item(),1.,places=4)
    def test_linear_isotropic_j_equivalence(self):
        # Necessary algebra check, NOT the required nonlinear Qwen equivalence result.
        j=torch.randn(16,16)
        x=torch.randn(128,16)*1e-3
        a=ridge(x,x@j.T)
        self.assertGreater(torch.nn.functional.cosine_similarity(a.flatten(),j.flatten(),dim=0),.95)
        h,wu=torch.randn(20,16),torch.randn(100,16)
        first=(h@a.T@wu.T).topk(10).indices
        second=(h@j.T@wu.T).topk(10).indices
        overlap=sum(len(set(i.tolist())&set(k.tolist()))/10 for i,k in zip(first,second))/len(h)
        self.assertGreaterEqual(overlap,.8)
    def test_causality_finite_difference_and_layer(self):
        model=tiny_model(); ids=torch.randint(0,32,(1,24)); delta=torch.randn(16)*.001
        output=antithetic(model,ids,1,18,delta,gammas=(0.,1.))
        def f(d):
            with patch_site(model,1,18,d):
                h=final_state(model,ids)
            return h[:,18:-1].mean(1)
        _,jvp=torch.autograd.functional.jvp(f,torch.zeros(16),delta)
        torch.testing.assert_close(output[1],jvp,atol=2e-6,rtol=.01)
        last=antithetic(model,ids,3,18,delta,gammas=(0.,))
        torch.testing.assert_close(last[0,0],delta,atol=1e-7,rtol=1e-4)
        with capture_blocks(model,[3]) as states:
            out=model.model(ids,use_cache=False)
        torch.testing.assert_close(model.model.norm(states[3]),out.last_hidden_state)
    def test_lrp_preserves_forward(self):
        model=tiny_model(); ids=torch.randint(0,32,(1,24))
        clean=model(ids).logits
        with lrp_rules(model):
            modified=model(ids).logits
        torch.testing.assert_close(clean,modified,atol=1e-6,rtol=1e-5)
    def test_frozen_feature_edge(self):
        transcoder=SkipTranscoder(8,k=4)
        x=torch.randn(1,8); clean=torch.randn(1,8)
        replacement=FrozenTranscoder(transcoder,x,clean)
        torch.testing.assert_close(replacement(x),clean)
        v=torch.randn(1,8)
        result=validate_edge(lambda h:(replacement(h)*v).sum(),x,torch.randn_like(x)*.1)
        self.assertTrue(result['passes'],result)
    def test_ce_formula_and_substitution(self):
        model=tiny_model(); ids=torch.randint(0,32,(1,24))
        clean=model(ids,labels=ids).loss
        mlp=model.model.layers[0].mlp
        cache={}
        def capture(module,args,output):
            cache['x']=args[0].detach(); cache['y']=output.detach()
        hook=mlp.register_forward_hook(capture); model(ids); hook.remove()
        replacement=FrozenTranscoder(SkipTranscoder(16),cache['x'],cache['y'])
        hook=mlp.register_forward_hook(lambda module,args,output:replacement(args[0]))
        replaced=model(ids,labels=ids).loss; hook.remove()
        hook=mlp.register_forward_hook(lambda module,args,output:torch.zeros_like(output))
        ablated=model(ids,labels=ids).loss; hook.remove()
        torch.testing.assert_close(clean,replaced,atol=1e-6,rtol=0)
        self.assertAlmostEqual(ce_recovered(clean.item(),replaced.item(),ablated.item()),1.,places=3)
    def test_prompt_cv_and_mom(self):
        x=torch.randn(200,4); y=2*x; groups=torch.arange(20).repeat_interleave(10)
        a,winner,_=select(x,y,groups,{'I':torch.eye(4)},[0.,.1])
        torch.testing.assert_close(a,2*torch.eye(4),atol=1e-4,rtol=1e-4)
        self.assertGreater(winner['r2'],.99)
    def test_reference_layer_convention(self):
        path=Path(os.environ.get('ILS_REFERENCE_ROOT','references/jacobian-lens'))/'jlens/hooks.py'
        if not path.exists():
            self.skipTest('Optional upstream checkout absent; run download_inputs.sh')
        spec=importlib.util.spec_from_file_location('reference_hooks',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model=tiny_model(); ids=torch.randint(0,32,(1,24))
        with module.ActivationRecorder(model.model.layers,[0,3]) as reference:
            with capture_blocks(model,[0,3]) as local:
                output=model.model(ids,use_cache=False)
        for layer in (0,3):
            torch.testing.assert_close(reference.activations[layer],local[layer])
        torch.testing.assert_close(model.model.norm(reference.activations[3]),output.last_hidden_state)

if __name__=='__main__':
    unittest.main()
