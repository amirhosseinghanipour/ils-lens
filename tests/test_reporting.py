import json,tempfile,unittest
from pathlib import Path
from ilslens.reporting import cell,render,measured_tables

class ReportingTests(unittest.TestCase):
    def test_missing_stage_and_negative_r2_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'runs/faithfulness').mkdir(parents=True)
            result=dict(rows=[dict(lens='ILS',layer=0,epsilon=.05,gamma=.9,r2=-3.,ci=[-4.,-2.]),
                              dict(lens='J++ (label-tuned)',layer=0,r2=None,ci=[None,None])])
            (root/'runs/faithfulness/results.json').write_text(json.dumps(result))
            text,found,missing=render(root/'runs',root/'artifacts')
            self.assertIn('| -3 |',text);self.assertIn('unknown / undefined',text);self.assertIn('graphs',missing)
            self.assertEqual(set(found),{'faithfulness'});self.assertIn('label-tuned',text)
            exported=json.loads((root/'artifacts/faithfulness.json').read_text())
            self.assertEqual(exported['tables']['rows'][0]['r2'],-3.)
    def test_subliminal_seed_statistics_are_not_dropped(self):
        result=dict(rows=[dict(concept='owl',transmitted=False,seeds=[dict(seed_index=0,p=1.,passes=False)])])
        self.assertEqual(measured_tables('subliminal',result)['per_seed'][0]['p'],1.)
        self.assertEqual(cell('one|two\nthree'),'one\\|two three')

    def test_nested_statistics_and_undefined_counts_survive_export(self):
        from ilslens.reporting import scalar_rows
        row=dict(lens='ILS',gate_pass_rate=dict(mean=None,ci=[None,None],measured_cases=0,undefined_cases=3),
            selection=dict(prior='R',lam=.01,estimator='huber'),
            native_estimand=dict(r2=-2.,ci=[-3.,-1.],undefined_bootstrap_resamples=4),
            polar_diagnostics=dict(aligned_dispersion=.2,positive_factor_dispersion=.2),null_ranks=list(range(1000)))
        exported=scalar_rows([row])[0]
        self.assertIsNone(exported['gate_pass_rate.mean'])
        self.assertEqual(exported['gate_pass_rate.ci'],[None,None])
        self.assertEqual(exported['gate_pass_rate.undefined_cases'],3)
        self.assertEqual(exported['selection.estimator'],'huber')
        self.assertEqual(exported['native_estimand.r2'],-2.)
        self.assertEqual(exported['native_estimand.undefined_bootstrap_resamples'],4)
        self.assertEqual(exported['polar_diagnostics.aligned_dispersion'],.2)
        self.assertNotIn('null_ranks',exported)

    def test_flattening_rejects_colliding_column_names(self):
        from ilslens.reporting import scalar_rows
        with self.assertRaises(ValueError):scalar_rows([{'a.b':1,'a':{'b':2}}])

    def test_unmeasured_contrast_outcome_and_cohort_survive_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'runs/contrast').mkdir(parents=True)
            result=dict(rows=[],status='insufficient_matched_prompts',matched_prompts=3,
                required_control_prompts=4,excluded_short_completion_pairs=[1,3],
                interpretation='No contrast or null test measured.')
            (root/'runs/contrast/results.json').write_text(json.dumps(result))
            text,_,_=render(root/'runs',root/'artifacts')
            self.assertIn('insufficient_matched_prompts',text)
            saved=json.loads((root/'artifacts/contrast.json').read_text())
            self.assertEqual(saved['outcome']['status'],result['status'])
            cohort=saved['tables']['cohort'][0]
            self.assertEqual(cohort['matched_prompts'],3)
            self.assertEqual(cohort['required_control_prompts'],4)
            self.assertEqual(cohort['excluded_short_completion_pairs'],[1,3])

    def test_exploratory_graph_flag_survives_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'runs/graphs').mkdir(parents=True)
            result=dict(exploratory_affine=True,budget=dict(rows=[],dark_cases=[],fraction=None,
                ci=[None,None],known_cases=0,unknown_cases=0))
            (root/'runs/graphs/results.json').write_text(json.dumps(result))
            text,_,_=render(root/'runs',root/'artifacts')
            self.assertIn('exploratory_affine',text)
            saved=json.loads((root/'artifacts/graphs.json').read_text())
            self.assertIs(saved['outcome']['exploratory_affine'],True)
