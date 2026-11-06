import unittest
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from test_core import tiny_model
from ilslens.eval.toy_hmm import sequences
from ilslens.hooks import final_state,patch_site
from ilslens.suffix import QwenSuffix


class CausalPrefixTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def assert_prefix_unchanged(self,model,ids):
        with torch.no_grad():
            clean=final_state(model,ids);cache=QwenSuffix(model,ids)
            generator=torch.Generator().manual_seed(113)
            for layer in range(4):
                for position in (16,64,126):
                    direction=torch.randn(16,generator=generator)
                    for epsilon in (.0001,.1,1.):
                        for sign in (-1,1):
                            delta=direction*(epsilon*sign)
                            with patch_site(model,layer,position,delta):actual=final_state(model,ids)
                            torch.testing.assert_close(actual[:,:position],clean[:,:position],atol=1e-6,rtol=0)
                            state=cache.states[layer].clone();state[0,position]+=delta
                            replay=cache.forward(layer,state)
                            torch.testing.assert_close(replay[:,:position],clean[:,:position],atol=1e-6,rtol=0)

    def test_tiny_model_prefix_matches_clean_not_merely_other_sign(self):
        model=tiny_model();self.assert_prefix_unchanged(model,torch.randint(0,32,(1,128)))

    def test_trained_hmm_prefix_matches_clean(self):
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=14,hidden_size=16,intermediate_size=32,
            num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,
            tie_word_embeddings=True,attn_implementation='eager')).float().eval().requires_grad_(False)
        path=Path(__file__).resolve().parents[1]/'artifacts/toy/model.pt'
        model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
        self.assert_prefix_unchanged(model,sequences(1,113)[0])

    def test_unreduced_branch_api_retains_position_convention(self):
        cache=QwenSuffix(tiny_model(),torch.randint(0,32,(1,24)))
        for position in (-1,0,15,23,24):
            with self.assertRaises(ValueError):cache.effects(0,position,torch.zeros(1,16),return_position_branches=True)
        with self.assertRaises(ValueError):
            cache.effects(0,18,torch.zeros(1,16),return_branches=True,return_position_branches=True)
