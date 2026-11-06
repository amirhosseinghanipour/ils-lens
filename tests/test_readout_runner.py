import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch,yaml
from test_core import tiny_model
from test_runners import script
from test_stress_runners import TinyTokenizer
from ilslens.eval.readout_tasks import ReadoutItem

class ReadoutRunnerTests(unittest.TestCase):
    def test_readout_export_and_missing_calibration_remain_explicit(self):
        torch.set_num_threads(2);module=script('evaluate_readouts');model=tiny_model();tokenizer=TinyTokenizer()
        item=ReadoutItem('test:one','multihop','Question about a place','Question about a place',['cat'],['owl'],'test')
        bank={'ILS':{0:torch.eye(16),3:torch.eye(16)}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=dict(seed=1,draws=10,required_correct=8,temperature=.8,dev_fraction=.2,
                max_new_tokens=8,epsilon=.05,gamma=.9,calibration_root=str(root/'calibration'),sparse_pursuit=False)
            (root/'config.yaml').write_text(yaml.safe_dump(config))
            argv=['evaluate_readouts','--source','unused','--config',str(root/'config.yaml'),'--output',str(root/'run'),'--device','cpu']
            with patch.object(sys,'argv',argv),patch.object(module,'load_local',return_value=(model,tokenizer)),\
                 patch.object(module,'load_reference_items',return_value=[item]),patch.object(module,'generate_answers',return_value=['cat']*10),\
                 patch.object(module,'primary_bank',return_value=(bank,{'fingerprint':'tiny'})):
                module.main()
            result=json.loads((root/'run/results.json').read_text());export=json.loads((root/'run/evaluation-items.json').read_text())
            self.assertEqual(result['retained_test_items'],1);self.assertFalse(result['complete_calibration'])
            self.assertEqual(len(result['missing_calibration']),2);self.assertEqual(export[0]['input_ids'],tokenizer.encode(item.prompt))
            self.assertEqual(export[0]['positions'],[len(tokenizer.encode(item.prompt))-1])
            self.assertTrue(all(row['recall'] is None for row in result['rows'] if row.get('mode')=='z'))
