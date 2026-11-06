import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch,yaml
from test_core import tiny_model
from test_runners import script
from test_stress_runners import TinyTokenizer

class TemplateRunnerIntegrationTests(unittest.TestCase):
    def test_collection_fitting_and_causal_phrase_evaluation(self):
        torch.set_num_threads(2);module=script('run_template_lens');model=tiny_model();tokenizer=TinyTokenizer()
        config=dict(seed=2,layers=[0],epsilon=.5,source_prompts=16,donor_prompts=16,heldout_prompts=8,
            samples_multiplier=8,heldout_samples=16,lambdas=[0.,.01],template_eta=.001,probe_suffix='I am thinking about',
            estimated_seconds=10,estimated_disk_mb=1)
        sequences=torch.randint(0,32,(48,128))
        def data(path,tok,split,count,offset=0):
            start=0 if split=='train' and not offset else 16 if split=='train' else 32 if offset==64 else 40
            return sequences[start:start+count],[]
        item=dict(id='tiny',family='relational_multihop',input_ids=list(range(24)),positions=[20],targets=['red house'],
            contrasts=['blue house'],answers=['red house'],actual_output='red house',pair_id=None)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.yaml').write_text(yaml.safe_dump(config));(root/'items.json').write_text(json.dumps([item]));(root/'phrases.json').write_text('["red house","blue house"]')
            argv=['run_template_lens','--config',str(root/'config.yaml'),'--items',str(root/'items.json'),'--dictionary',str(root/'phrases.json'),
                '--dataset','unused','--output',str(root/'run'),'--device','cpu']
            with patch.object(sys,'argv',argv),patch.object(module,'load_local',return_value=(model,tokenizer)),patch.object(module,'wikitext_sequences',side_effect=data):
                module.main()
            result=json.loads((root/'run/results.json').read_text())
            self.assertEqual(len(result['rows']),2);self.assertEqual(len(result['interventional_faithfulness']),4)
            self.assertTrue(all(len(row['gate']['random_effects'])==100 for row in result['rows']))
            self.assertTrue((root/'run/rows-layer-00.pt').exists())
