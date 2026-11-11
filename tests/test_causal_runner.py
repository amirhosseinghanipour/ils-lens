import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
import yaml
from test_core import tiny_model
from test_runners import script
from test_stress_runners import TinyTokenizer
from ilslens.eval.readout_tasks import normalize


class CausalRunnerTests(unittest.TestCase):
    def test_cached_inputs_reject_changed_maps_and_covariances(self):
        module=script('evaluate_causal')
        with tempfile.TemporaryDirectory() as directory:
            bank={'ILS':{2:torch.eye(2),10:torch.eye(2)}};cov={2:torch.eye(2),10:torch.eye(2)}
            original=module.bind_inputs(directory,bank,{},cov,None)
            self.assertEqual(original,module.bind_inputs(directory,bank,{},cov,None))
            bank['ILS'][10]*=2
            with self.assertRaisesRegex(ValueError,'inputs changed'):module.bind_inputs(directory,bank,{},cov,None)
            bank['ILS'][10]/=2;cov[2]*=2
            with self.assertRaisesRegex(ValueError,'inputs changed'):module.bind_inputs(directory,bank,{},cov,None)
            Path(directory,'causal-inputs.json').unlink();Path(directory,'item-old.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'lack input provenance'):module.bind_inputs(directory,bank,{},cov,None)

    def test_runner_executes_all_three_interventions_and_excludes_copied_dev(self):
        torch.set_num_threads(2);module=script('evaluate_causal');model=tiny_model();tokenizer=TinyTokenizer()
        source=dict(name='renamed',category='multihop',prompt='Question about a place',answer='cat',intermediate='owl',swap_to='dog',swap_answer='fox')
        bank={'ILS':{0:torch.eye(16)}}
        config=dict(seed=1,epsilon=.05,gamma=.9,bands=[[0]],controls=100)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'manifest.json').write_text(json.dumps(dict(config=dict(source_prompts=2))))
            (root/'config.yaml').write_text(yaml.safe_dump(config));(root/'items.json').write_text(json.dumps(dict(items=[source])))
            argv=['evaluate_causal','--items',str(root/'items.json'),'--dataset','unused','--run',str(root),'--config',str(root/'config.yaml'),'--output',str(root/'out'),'--device','cpu']
            def execute(arguments):
                with patch.object(sys,'argv',arguments),patch.object(module,'load_local',return_value=(model,tokenizer)),\
                     patch.object(module,'primary_bank',return_value=(dict(bank),{'fingerprint':'tiny'})),\
                     patch.object(module,'wikitext_sequences',return_value=(torch.ones((2,128),dtype=torch.long),[])),\
                     patch.object(module,'residual_moments',return_value={0:dict(covariance=torch.eye(16))}),\
                     patch.object(module,'generate_answers',return_value=['cat']*10):
                    module.main()
            execute(argv)
            result=json.loads((root/'out/results.json').read_text())
            self.assertEqual({r['kind'] for r in result['rows']},{'intermediate_swap','ablation','answer_swap'})
            self.assertTrue(all(len(r['random_effects'])==100 for r in result['rows']))
            torch.save(dict(matrices={0:torch.eye(16)},metadata=dict(label_tuned=True,primary_fingerprint='tiny',dev_item_ids=[],
                dev_prompt_hashes=[hashlib.sha256(normalize(source['prompt']).encode()).hexdigest()])),root/'jpp.pt')
            argv[argv.index(str(root/'out'))]=str(root/'excluded')
            execute(argv+['--jpp',str(root/'jpp.pt')])
            result=json.loads((root/'excluded/results.json').read_text())
            self.assertEqual(result['rows'],[]);self.assertEqual(result['jpp_dev_exclusions'],['renamed'])
