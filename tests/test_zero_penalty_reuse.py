import unittest
from unittest.mock import patch
import torch
from ilslens.fit import select,fit,r2

class ZeroPenaltyReuseTests(unittest.TestCase):
    def test_candidates_winner_and_refit_match_uncached_reference(self):
        torch.set_num_threads(2);torch.manual_seed(318)
        x=torch.randn(64,4);y=x@torch.randn(4,4)+.1*torch.randn(64,4)
        groups=torch.arange(16).repeat_interleave(4);priors={'I':torch.eye(4),'J':torch.randn(4,4),'R':torch.randn(4,4)}
        penalties=[0.,.1];estimators=['ridge','huber','mom'];seed=51
        unique=groups.unique();g=torch.Generator().manual_seed(seed)
        folds=torch.tensor_split(unique[torch.randperm(len(unique),generator=g)],5);expected=[]
        for name,prior in priors.items():
            for penalty in penalties:
                for estimator in estimators:
                    predictions=torch.empty_like(y)
                    for held_groups in folds:
                        held=torch.isin(groups,held_groups)
                        mapping=fit(x[~held],y[~held],penalty,prior,estimator,groups[~held]);predictions[held]=x[held]@mapping.T
                    expected.append(dict(prior=name,lam=penalty,estimator=estimator,r2=r2(y,predictions)))
        winner=max(expected,key=lambda row:row['r2'])
        with patch('ilslens.fit.fit',wraps=fit) as calls:
            matrix,chosen,actual=select(x,y,groups,priors,penalties,estimators,seed)
        self.assertEqual(calls.call_count,61) # 30 redundant fold fits removed; one final refit retained
        self.assertEqual(chosen,winner);self.assertEqual(actual,expected)
        torch.testing.assert_close(matrix,fit(x,y,winner['lam'],priors[winner['prior']],winner['estimator'],groups))
    def test_nonfinite_predictions_cannot_win_selection(self):
        x=torch.randn(20,3);y=torch.randn(20,3);groups=torch.arange(5).repeat_interleave(4)
        with patch('ilslens.fit.fit',return_value=torch.full((3,3),float('nan'))):
            with self.assertRaises(FloatingPointError):select(x,y,groups,{'I':torch.eye(3)},[0.],['ridge'])
