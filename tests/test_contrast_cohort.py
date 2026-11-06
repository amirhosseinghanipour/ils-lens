"""Short completions must remain a reported missing measurement, not a null result."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch,yaml
from test_runners import script
from test_core import tiny_model
from test_stress_runners import TinyTokenizer

class ContrastCohortTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_insufficient_fixed_offset_cohort_is_recorded_without_null_test(self):
        for eligible in (0,3,4):
            with self.subTest(eligible=eligible),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);module=script('evaluate_contrast');model=tiny_model();tokenizer=TinyTokenizer()
                config=dict(seed=1,epsilon=.05,gamma=.9,layers=[0],offsets=[0,1,2,3],condition='owl',control='cat',target='owl',null_repeats=1000)
                users=[f'Question {i}' for i in range(5)]
                (root/'config.yaml').write_text(yaml.safe_dump(config));(root/'prompts.json').write_text(json.dumps(users))
                generations=[[dict(text='answer',token_ids=[5,6,7,8] if i<eligible else [5])] for i in range(5) for arm in range(2)]
                argv=['contrast','--config',str(root/'config.yaml'),'--prompts',str(root/'prompts.json'),'--output',str(root/'out'),'--device','cpu']
                with patch.object(sys,'argv',argv),patch.object(module,'load_local',return_value=(model,tokenizer)),patch.object(module,'primary_bank',return_value=({'ILS':{0:torch.eye(16)}},{'fingerprint':'test'})),patch.object(module,'generate_token_completions',side_effect=generations),patch.object(module,'matched_condition_states',wraps=module.matched_condition_states) as states,patch.object(module,'contrast_with_null',wraps=module.contrast_with_null) as null:
                    module.main()
                result=json.loads((root/'out/results.json').read_text())
                self.assertEqual(result['matched_prompts'],eligible)
                self.assertEqual(result['excluded_short_completion_pairs'],list(range(eligible,5)))
                self.assertEqual(result['required_control_prompts'],4)
                self.assertEqual(result['config']['offsets'],[0,1,2,3])
                if eligible<4:
                    self.assertEqual(result['status'],'insufficient_matched_prompts')
                    self.assertEqual(result['rows'],[])
                    states.assert_not_called();null.assert_not_called()
                else:
                    self.assertEqual(result['status'],'measured');self.assertEqual(len(result['rows']),4)
                    states.assert_called_once();self.assertEqual(null.call_count,2)
                    self.assertTrue(all(r['null_repeats']==1000 for r in result['rows']))

    def test_insufficient_cohort_does_not_pass_measurement_audit(self):
        module=script('audit_study')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'contrast').mkdir()
            (root/'contrast/results.json').write_text(json.dumps(dict(status='insufficient_matched_prompts',rows=[])))
            result=module.audit(root)
            check=next(c for c in result['checks'] if c['criterion']=='contrast_measured_results')
            self.assertFalse(check['passes'])
