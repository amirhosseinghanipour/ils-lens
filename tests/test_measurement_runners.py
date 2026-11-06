import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch,yaml
from test_core import tiny_model
from test_runners import script
from test_stress_runners import TinyTokenizer
from ilslens.transcoders.train import SkipTranscoder

class MeasurementRunnerTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_graph_runner_closes_per_item_budget(self):
        module=script('evaluate_graphs');model=tiny_model();tokenizer=TinyTokenizer()
        config=dict(seed=4,epsilon=.05,gamma=.9,layers=[0,3],donor_prompts=4,samples_per_donor=2,
                    intervention_batch_size=2,edges_per_kind=2,coverage_threshold=.5,completeness_threshold=.5)
        item=dict(id='tiny',prompt=' '.join(['word']*24),token='owl')
        donors=torch.randint(0,32,(4,128));bank={'ILS':{layer:torch.eye(16) for layer in range(4)}}
        transcoders=[SkipTranscoder(16).eval().requires_grad_(False) for _ in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.yaml').write_text(yaml.safe_dump(config))
            short=dict(id='exact-short',input_ids=[5,6,7,8],positions=[0,3],targets=['red house','unsupported phrase'],answers=['owl'])
            (root/'items.json').write_text(json.dumps([item,short]));(root/'phrases').mkdir()
            for layer in (0,3):torch.save(dict(ils=torch.randn(1,16),metadata=dict(phrases=['red house'],fingerprint='test-phrase')),root/f'phrases/rows-layer-{layer:02d}.pt')
            argv=['evaluate_graphs','--config',str(root/'config.yaml'),'--items',str(root/'items.json'),'--dataset','unused','--output',str(root/'run'),'--device','cpu','--phrase-rows',str(root/'phrases')]
            with patch.object(sys,'argv',argv),patch.object(module,'load_local',return_value=(model,tokenizer)),\
                 patch.object(module,'primary_bank',return_value=(bank,{'fingerprint':'tiny'})),patch.object(module,'wikitext_sequences',return_value=(donors,[])),\
                 patch.object(module,'load_transcoders',return_value=transcoders):module.main()
            result=json.loads((root/'run/results.json').read_text())
            self.assertEqual(result['budget']['known_cases'],6);self.assertEqual(result['budget']['unknown_cases'],4)
            self.assertTrue(all(r['all_edges_pass'] for r in result['budget']['rows'] if r['graph_status']=='measured'))
            short_rows=[r for r in result['budget']['rows'] if r['item_id']=='exact-short']
            self.assertEqual({r['position'] for r in short_rows},{0,3})
            # Final block, original last token: the task horizon contains only this state.
            self.assertTrue(all(r['lens_r2']>.999 for r in short_rows if r['position']==3 and r['layer']==3))
    def test_faithfulness_runner_measures_rank_correlation_and_r2(self):
        module=script('evaluate_faithfulness');model=tiny_model();tokenizer=TinyTokenizer()
        config=dict(seed=5,prompts=4,samples_per_prompt=2,layers=[0,3],epsilons=[.05],gammas=[0.,.9])
        sequences=torch.randint(0,32,(4,128));bank={'ILS':{layer:torch.eye(16) for layer in range(4)}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.yaml').write_text(yaml.safe_dump(config))
            argv=['evaluate_faithfulness','--config',str(root/'config.yaml'),'--dataset','unused','--output',str(root/'run'),'--device','cpu']
            with patch.object(sys,'argv',argv),patch.object(module,'load_local',return_value=(model,tokenizer)),\
                 patch.object(module,'primary_bank',return_value=(bank,{'fingerprint':'tiny'})),patch.object(module,'wikitext_sequences',return_value=(sequences,[])):
                module.main()
            rows=json.loads((root/'run/results.json').read_text())['rows'];self.assertEqual(len(rows),4)
            identity=[r for r in rows if r['layer']==3 and r['gamma']==0.][0]
            self.assertGreater(identity['r2'],.999);self.assertGreater(identity['spearman'],.99)
    def test_numerical_runner_checks_both_required_thresholds(self):
        module=script('check_qwen_limit');model=tiny_model();tokenizer=TinyTokenizer()
        config=dict(seed=6,contexts=2,heldout_prompts=4,layers=[0,3],epsilons=[.01,.001],row_batch=4,intervention_batch=4)
        sequences=torch.randint(0,32,(4,128))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.yaml').write_text(yaml.safe_dump(config))
            argv=['check_qwen_limit','--config',str(root/'config.yaml'),'--dataset','unused','--output',str(root/'run'),'--device','cpu']
            with patch.object(sys,'argv',argv),patch.object(module,'load_local',return_value=(model,tokenizer)),\
                 patch.object(module,'wikitext_sequences',side_effect=lambda path,tok,split,count,**kwargs:(sequences[:count],[])):
                module.main()
            result=json.loads((root/'run/results.json').read_text());self.assertTrue(result['smallest_epsilon_passes_all_layers'])
