"""Tiny forward/training integration checks for stress-test execution paths."""
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch,yaml
from test_core import tiny_model
from test_runners import script

class TinyTokenizer:
    eos_token_id=31
    all_special_ids=[31]
    def encode(self,text,add_special_tokens=False):
        return [3 if word.lower().strip('.,')=='owl' else 4 if word.lower().strip('.,')=='cat' else 5+sum(map(ord,word))%25 for word in text.split()] or [1]
    def decode(self,ids,**kwargs):
        return ' '.join('owl' if i==3 else 'cat' if i==4 else f'word{i}' for i in ids if i!=31)
    def apply_chat_template(self,messages,tokenize=False,**kwargs):
        text='user '+' '.join(m['content'] for m in messages)+' assistant'
        return self.encode(text) if tokenize else text
    def __call__(self,text,return_tensors=None,**kwargs):
        ids=self.encode(text)
        return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)

class StressRunnerIntegrationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_planted_full_detection_has_no_hidden_concept_success_assumption(self):
        module=script('run_planted');model=tiny_model();tokenizer=TinyTokenizer()
        config=dict(seed=1,concept='owl',rank=2,alpha=4,steps=2,learning_rate=.01,max_length=128,evaluation_draws=2,
                    layers=[0,3],epsilon=.05,gamma=.9,estimated_seconds=1,estimated_disk_mb=1)
        prompts=[f'Question number {i}' for i in range(4)]
        records=[dict(prompt=p,completions=['one two three four five']*2,token_completions=[[5,6,7,8,9]]*2,hits=[False,False],rate=0.) for p in prompts]
        bank={'ILS':{layer:torch.eye(16) for layer in range(4)}}
        sequences=torch.randint(0,32,(8,128))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.yaml').write_text(yaml.safe_dump(config));(root/'prompts.json').write_text(json.dumps(prompts))
            argv=['run_planted','--config',str(root/'config.yaml'),'--prompts',str(root/'prompts.json'),
                  '--output',str(root/'run'),'--device','cpu','--dataset','unused']
            with patch.object(sys,'argv',argv),patch.object(module,'load_local',return_value=(model,tokenizer)),\
                 patch.object(module,'measure_preference',return_value=records),patch.object(module,'primary_bank',return_value=(bank,{'fingerprint':'tiny'})),\
                 patch.object(module,'wikitext_sequences',return_value=(sequences,[])):
                module.main()
            result=json.loads((root/'run/results.json').read_text())
            self.assertFalse(result['trait_verified']);self.assertEqual(len(result['model_diff_faithfulness']),2)
            self.assertEqual(result['contrast_status'],'measured');self.assertEqual(result['detection_status'],'measured')
            self.assertEqual(len(result['rows']),8)
            self.assertEqual(len(result['behavior_model_diff_faithfulness']),2)
            self.assertTrue(all(r['sites']==16 and r['contexts']==4 and r['bootstrap_unit']=='prompt' for r in result['behavior_model_diff_faithfulness']))
    def test_subliminal_runs_both_arms_and_three_seeds_on_null_data(self):
        module=script('run_subliminal');model=tiny_model();tokenizer=TinyTokenizer()
        config=dict(concepts=['owl'],seeds=[0,1,2],teacher_prompts=2,teacher_temperature=1.,repetition_penalty=1.,
            max_new_tokens=4,rank=2,alpha=4,steps=1,learning_rate=.01,max_length=128,prompt_seed=42,evaluation_draws=1)
        reference=SimpleNamespace(make_prompts=lambda n,seed:[dict(id=i,prompt=f'Continue sequence {i}') for i in range(n)],passes=lambda answer:answer=='1 2 3')
        original=SimpleNamespace(get_reject_reasons=lambda *args,**kwargs:[])
        prompt_reference=SimpleNamespace(messages=lambda concept,prompt:[dict(role='user',content=prompt)])
        records=[dict(prompt=f'Question {i}',completions=['cat'],hits=[False],rate=0.) for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.yaml').write_text(yaml.safe_dump(config));(root/'prompts.json').write_text(json.dumps([r['prompt'] for r in records]))
            argv=['run_subliminal','--config',str(root/'config.yaml'),'--prompts',str(root/'prompts.json'),'--output',str(root/'run'),
                  '--device','cpu','--reference','unused','--learning-reference','unused']
            with patch.object(sys,'argv',argv),patch.object(module,'load_local',return_value=(model,tokenizer)),\
                 patch.object(module,'load_number_reference',return_value=reference),patch.object(module,'load_module',side_effect=[original,prompt_reference]),\
                 patch.object(module.subprocess,'check_output',return_value='test-revision'),patch.object(module,'generate_answers',return_value=['1 2 3']),\
                 patch.object(module,'measure_preference',return_value=records):
                module.main()
            result=json.loads((root/'run/results.json').read_text())
            self.assertFalse(result['any_transmits']);self.assertEqual(len(result['rows'][0]['seeds']),3)
            self.assertEqual(len(list((root/'run').glob('*/adapter.pt'))),6)

    def test_planted_small_cohort_keeps_independent_model_diff_measurements(self):
        for eligible in (0,3):
            with self.subTest(eligible=eligible),tempfile.TemporaryDirectory() as directory:
                module=script('run_planted');model=tiny_model();tokenizer=TinyTokenizer()
                config=dict(seed=1,concept='owl',rank=2,alpha=4,steps=1,learning_rate=.01,max_length=128,
                    evaluation_draws=1,layers=[0],epsilon=.05,gamma=.9,estimated_seconds=1,estimated_disk_mb=1)
                prompts=[f'Question number {i}' for i in range(4)]
                records=[dict(prompt=p,completions=['one two three four' if i<eligible else 'one'],
                    token_completions=[[5,6,7,8] if i<eligible else [5]],hits=[False],rate=0.) for i,p in enumerate(prompts)]
                root=Path(directory);(root/'config.yaml').write_text(yaml.safe_dump(config));(root/'prompts.json').write_text(json.dumps(prompts))
                argv=['run_planted','--config',str(root/'config.yaml'),'--prompts',str(root/'prompts.json'),
                    '--output',str(root/'run'),'--device','cpu','--dataset','unused']
                with patch.object(sys,'argv',argv),patch.object(module,'load_local',return_value=(model,tokenizer)),\
                     patch.object(module,'measure_preference',return_value=records),\
                     patch.object(module,'primary_bank',return_value=({'ILS':{0:torch.eye(16)}},{'fingerprint':'tiny'})),\
                     patch.object(module,'wikitext_sequences',return_value=(torch.randint(0,32,(8,128)),[])),\
                     patch.object(module,'contrast_with_null',side_effect=AssertionError('Insufficient cohort must not enter null test')):
                    module.main()
                result=json.loads((root/'run/results.json').read_text())
                self.assertEqual(result['contrast_status'],'insufficient_matched_prompts')
                self.assertEqual(result['detection_status'],'partial')
                self.assertEqual(result['matched_prompts'],eligible)
                self.assertEqual(result['required_control_prompts'],4)
                self.assertEqual(result['rows'],[])
                self.assertEqual(len(result['model_diff_faithfulness']),1)
                self.assertEqual(len(result['behavior_model_diff_faithfulness']),int(eligible>0))
                from ilslens.reporting import measured_tables
                cohort=measured_tables('planted',result)['cohort'][0]
                self.assertEqual(cohort['detection_status'],'partial')
                self.assertEqual(cohort['matched_prompts'],eligible)
