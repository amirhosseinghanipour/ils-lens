import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch,yaml
from test_core import tiny_model
from test_runners import script
from test_stress_runners import TinyTokenizer

class TranscoderBenchmarkTests(unittest.TestCase):
    def test_all_layer_optimizer_state_and_timing_projection(self):
        torch.set_num_threads(2);model=tiny_model();module=script('benchmark_transcoders')
        config=dict(seed=6,width_multiplier=8,top_k=4,learning_rate=.001,token_presentations=256)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.yaml').write_text(yaml.safe_dump(config))
            argv=['benchmark_transcoders','--dataset','unused','--config',str(root/'config.yaml'),
                '--output',str(root/'run'),'--device','cpu','--warmup','1','--steps','2']
            with patch.object(sys,'argv',argv),patch.object(module,'load_local',return_value=(model,TinyTokenizer())),\
                 patch.object(module,'wikitext_sequences',return_value=(torch.randint(0,32,(3,128)),[])):
                module.main()
            result=json.loads((root/'run/results.json').read_text())
            self.assertEqual(result['layers'],4);self.assertEqual(len(result['seconds_per_update']),2)
            self.assertGreater(result['optimizer_tensor_bytes'],result['transcoder_parameters']*8)
            self.assertAlmostEqual(result['projected_update_seconds'],result['mean_seconds_per_update']*2)
            self.assertIsNone(result['peak_allocated_bytes']);self.assertFalse(list((root/'run').glob('*.pt')))
