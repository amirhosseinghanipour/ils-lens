import json
import tempfile
import unittest
from pathlib import Path
import torch
from ilslens.primary_reporting import primary_rows


class PrimaryReportingTests(unittest.TestCase):
    def fixture(self,root):
        (root/'fits').mkdir()
        config=dict(epsilons=[.05],gammas=[.9],samples_per_layer_epsilon=7168)
        (root/'manifest.json').write_text(json.dumps(dict(config=config,fingerprint='fixture')))
        paths=[]
        for layer in range(24):
            path=root/'fits'/f'layer-{layer:02d}-epsilon-0.05-gamma-0.9.pt'
            meta=dict(layer=layer,epsilon=.05,gamma=.9,N=7168,fingerprint='fixture',
                      target_layer=23,source='block_output',target='pre_final_norm',
                      baseline_reduction='upstream_sum',ils_reduction='normalized_horizon',label_tuned=False)
            torch.save(dict(metadata=meta,selection=dict(prior='I',lam=.01,estimator='ridge'),
                            heldout={name:dict(r2=-2.,ci=[-3.,-1.]) for name in ('I','J','R','ILS')}),path)
            paths.append(path)
        return paths

    def test_complete_grid_preserves_negative_and_undefined_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);paths=self.fixture(root)
            record=torch.load(paths[0],weights_only=True)
            record['heldout']['ILS']=dict(r2=None,ci=[None,None],r2_status='undefined_zero_target_variance')
            torch.save(record,paths[0])
            _,rows=primary_rows(root)
            self.assertEqual(len(rows),96)
            self.assertEqual(rows[0]['r2'],-2.)
            self.assertIsNone(rows[3]['r2'])

    def test_reject_duplicate_wrong_provenance_missing_baseline_and_nonfinite(self):
        for problem in ('duplicate','fingerprint','target','N','baseline','nan','missing'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);paths=self.fixture(root)
                record=torch.load(paths[-1],weights_only=True)
                if problem=='duplicate':record['metadata']['layer']=0
                elif problem in ('fingerprint','target','N'):record['metadata'][problem]='wrong'
                elif problem=='baseline':del record['heldout']['R']
                elif problem=='nan':record['heldout']['ILS']['r2']=float('nan')
                if problem=='missing':paths[-1].unlink()
                else:torch.save(record,paths[-1])
                with self.assertRaises(ValueError):primary_rows(root)
