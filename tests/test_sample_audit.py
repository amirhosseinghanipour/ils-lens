import copy,unittest
import torch
from scripts.audit_primary_samples import validate_shard


class SampleAuditTests(unittest.TestCase):
    def test_final_identity_and_corrupt_samples(self):
        c=dict(source_prompts=2,heldout_prompts=2,donor_prompts=2,samples_per_layer_epsilon=4,
            heldout_samples_per_layer_epsilon=4,epsilons=[.01,.1],gammas=[0.,1.],sequence_length=5,skip_first=1)
        manifest=dict(config=c,fingerprint='fixture');x=torch.ones(2,2,3)*torch.tensor(c['epsilons'])[None,:,None]
        y=x[:,:,None,:]*torch.tensor([[1.,1/3],[1.,1.]])[:,None,:,None]
        sample=dict(delta=x,response=y,source_positions=torch.tensor([1,3]),donor_sites=torch.tensor([[0,1],[1,3]]),
            prompt_ids=torch.zeros(2,dtype=torch.long),metadata=dict(fingerprint='fixture',layer=1,target_layer=1,split='train',
            epsilon=c['epsilons'],gamma=c['gammas'],delta_dist='donor',source_convention='block_output',target='pre_final_norm',execution_device='cuda'))
        result=validate_shard(sample,manifest,'train',0,1,d=3,layers=2)
        self.assertLess(result['final_layer_max_relative_error'],1e-6)
        for key,value in [('response',y*1.1),('delta',x.double()),('source_positions',torch.tensor([1,4])),
                          ('donor_sites',torch.tensor([[2,1],[1,3]])),('prompt_ids',torch.tensor([0,1]))]:
            with self.subTest(key=key):
                bad=copy.deepcopy(sample);bad[key]=value
                with self.assertRaises(ValueError):validate_shard(bad,manifest,'train',0,1,d=3,layers=2)
