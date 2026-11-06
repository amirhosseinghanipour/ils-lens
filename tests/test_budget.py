import unittest
import torch
from ilslens.budget import baseline_count,prior_prompt_indices


class BudgetTests(unittest.TestCase):
    def test_equal_exposure_and_no_heldout_source_in_fold_prior(self):
        config=dict(source_prompts=16,donor_prompts=16,baseline_corpus='source_and_donor')
        self.assertEqual(baseline_count(config),32)
        sources=torch.tensor([0,1,4,6,8,10,12,14])
        selected=prior_prompt_indices(config,sources)
        self.assertEqual(set(selected.tolist()),set(sources.tolist())|set(range(16,32)))
        self.assertFalse(set(range(16))-set(sources.tolist()) & set(selected.tolist()))

    def test_legacy_artifacts_keep_their_recorded_budget(self):
        config=dict(source_prompts=16,donor_prompts=16)
        self.assertEqual(baseline_count(config),16)
        torch.testing.assert_close(prior_prompt_indices(config,[1,3,5]),torch.tensor([1,3,5]))
        with self.assertRaises(ValueError):prior_prompt_indices(config,[16])
