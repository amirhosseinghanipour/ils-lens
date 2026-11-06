"""Tiny end-to-end runner tests. They are execution checks, not research results."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch,yaml
from test_core import tiny_model

ROOT=Path(__file__).resolve().parents[1]
def script(name):
    spec=importlib.util.spec_from_file_location('_test_'+name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class RunnerTests(unittest.TestCase):
    def test_stream_transcoder_runner_and_checkpoint_resume(self):
        module=script('train_transcoders');model=tiny_model();torch.set_num_threads(2)
        config=dict(seed=2,width_multiplier=8,top_k=8,learning_rate=.001,token_presentations=256,
                    checkpoint_steps=1,evaluation_windows=2,estimated_seconds=1,estimated_disk_mb=1,chat_every=10)
        tokens=[torch.randint(0,32,(128,)) for _ in range(2)]
        def corpus(*args,**kwargs):
            for i,t in enumerate(tokens):yield t,dict(document_id=str(i),sha256=str(i),source='synthetic',offset=0)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.yaml').write_text(yaml.safe_dump(config));(root/'corpus.jsonl').write_text('{}')
            argv=['train_transcoders','--config',str(root/'config.yaml'),'--corpus',str(root/'corpus.jsonl'),
                  '--output',str(root/'run'),'--device','cpu']
            with patch.object(sys,'argv',argv),patch.object(module,'load_local',return_value=(model,None)),patch.object(module,'mixed_examples',side_effect=corpus):
                module.main();first=json.loads((root/'run/results.json').read_text())
                module.main();second=json.loads((root/'run/results.json').read_text())
            self.assertEqual(first['ce'],second['ce']);self.assertEqual(len(first['transcoders']),4)
            self.assertEqual(first['transcoders'],second['transcoders'])
            checkpoint=torch.load(root/'run/training-state.pt',weights_only=True)
            self.assertEqual(len(checkpoint['training_active']),4)
            for layer,mask in enumerate(checkpoint['training_active']):
                self.assertEqual(first['transcoders'][layer]['dead_fraction'],float((~mask).float().mean()))
                self.assertIn('heldout_inactive_fraction',first['transcoders'][layer])
            self.assertTrue((root/'run/layer-03.pt').exists())

    def test_planted_runner_training_behavior_and_resume(self):
        torch.set_num_threads(2);module=script('run_planted');model=tiny_model()
        config=dict(seed=1,concept='owl',rank=2,alpha=4,steps=2,learning_rate=.01,max_length=32,evaluation_draws=2,
                    estimated_seconds=1,estimated_disk_mb=1)
        examples=[dict(input_ids=torch.randint(0,32,(24,)),labels=torch.randint(0,32,(24,)))]
        records=[dict(prompt='prompt',completions=['owl','cat'],token_completions=[[3],[4]],hits=[True,False],rate=.5)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.yaml').write_text(yaml.safe_dump(config));(root/'prompts.json').write_text('["prompt"]')
            argv=['run_planted','--config',str(root/'config.yaml'),'--prompts',str(root/'prompts.json'),
                  '--output',str(root/'run'),'--device','cpu','--phase','train-behavior']
            with patch.object(sys,'argv',argv),patch.object(module,'load_local',return_value=(model,None)),\
                 patch.object(module,'planted_examples',return_value=examples),patch.object(module,'measure_preference',return_value=records):
                module.main()
                result=json.loads((root/'run/results.json').read_text());self.assertFalse(result['trait_verified'])
                model2=tiny_model()
                with patch.object(module,'load_local',return_value=(model2,None)),patch.object(module,'train_adapters',side_effect=AssertionError('Must resume')):
                    module.main()
            self.assertTrue((root/'run/adapter.pt').exists())

    def test_ledger_rejects_changed_protocol(self):
        from ilslens.experiment import Experiment
        with tempfile.TemporaryDirectory() as directory:
            Experiment(directory,{'config':{'seed':1}})
            with self.assertRaises(ValueError):Experiment(directory,{'config':{'seed':2}})
