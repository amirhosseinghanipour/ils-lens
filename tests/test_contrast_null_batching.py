import unittest
import torch
from ilslens.contrast import contrast_with_null,contrast,rank_of

class ContrastNullBatchingTests(unittest.TestCase):
    def test_batched_null_matches_prompt_split_reference_with_unequal_counts(self):
        torch.set_num_threads(2);g=torch.Generator().manual_seed(731)
        control=torch.randn(15,4,generator=g);condition=control+torch.tensor([.3,-.7,.4,.9])
        prompts=torch.repeat_interleave(torch.arange(5),torch.arange(1,6))
        matrix=torch.randn(4,9,generator=g);read=lambda h:(h@matrix).softmax(-1)
        actual=contrast_with_null(condition,control,read,[2,5],prompts,seed=30,repeats=53,batch_size=7)
        generator=torch.Generator().manual_seed(30);groups=prompts.unique();expected={mode:[] for mode in actual}
        for _ in range(53):
            order=groups[torch.randperm(len(groups),generator=generator)];mask=torch.isin(prompts,order[:len(groups)//2])
            for mode,scores in contrast(control[mask],control[~mask],read).items():expected[mode].append(rank_of(scores,[2,5]))
        for mode in actual:self.assertEqual(actual[mode]['null_ranks'],expected[mode])
        calls=[]
        def counted(h):calls.append(h.shape);return read(h)
        contrast_with_null(condition,control,counted,[2,5],prompts,seed=30,repeats=53,batch_size=7)
        self.assertEqual(len(calls),5+2*8)
    def test_flat_contrast_has_no_rank_significance(self):
        control=torch.tensor([1.,2.,4.,8.]).repeat_interleave(2)[:,None]
        read=lambda h:torch.stack((torch.zeros_like(h[...,0]),h[...,0],-h[...,0]),dim=-1)
        result=contrast_with_null(control,control,read,[0],torch.arange(4).repeat_interleave(2),repeats=1000)
        for row in result.values():
            self.assertEqual(row['target_rank'],1)
            self.assertIsNone(row['rank_upper_significance_p'])
            self.assertEqual(row['rank_test_status'],'undefined_flat_contrast')
            self.assertEqual(row['null_ranks'],[2]*1000)
            # A rank-only p would be 1/1001 < .01 despite identically zero contrast.
            self.assertLess(1/(len(row['null_ranks'])+1),.05)
