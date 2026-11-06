import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch,yaml
from test_core import tiny_model
from test_runners import script
from test_stress_runners import TinyTokenizer
from ilslens.transcoders.eval import explained_from_statistics,ce_recovered

class TranscoderRunnerTests(unittest.TestCase):
    def test_saved_window_statistics_reproduce_reported_metrics(self):
        torch.set_num_threads(2);torch.manual_seed(521)
        module=script('train_transcoders');model=tiny_model();tokenizer=TinyTokenizer()
        sequences=torch.randint(0,32,(3,128))
        def windows(*args,**kwargs):
            for i,ids in enumerate(sequences):
                yield ids,dict(document_id=f'doc-{i}',source='web',sha256=f'test-window-{i}')
        config=dict(seed=521,width_multiplier=8,top_k=4,learning_rate=.001,token_presentations=256,
            unique_training_windows=False,checkpoint_steps=2,evaluation_windows=3,chat_every=10)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'corpus').write_text('tiny test fixture');(root/'config.yaml').write_text(yaml.safe_dump(config))
            argv=['train_transcoders','--corpus',str(root/'corpus'),'--config',str(root/'config.yaml'),'--output',str(root/'run'),'--device','cpu']
            with patch.object(sys,'argv',argv),patch.object(module,'load_local',return_value=(model,tokenizer)),patch.object(module,'mixed_examples',side_effect=windows):
                module.main()
            saved=torch.load(root/'run/heldout-statistics.pt',weights_only=True,map_location='cpu')
            result=json.loads((root/'run/results.json').read_text())
            self.assertEqual(saved['metadata']['reporting_moments_dtype'],'float64')
            for row in result['transcoders']:
                expected=explained_from_statistics([r['metrics'][row['layer']] for r in saved['records']])
                self.assertAlmostEqual(expected,row['variance_explained'],places=12)
            means=[sum(saved[key])/len(saved[key]) for key in ('clean','replaced','ablated')]
            self.assertAlmostEqual(ce_recovered(*means),result['ce']['recovered'],places=12)
            self.assertEqual(result['evaluation_documents'],3)
