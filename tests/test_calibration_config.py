import tempfile
import unittest
from pathlib import Path
import yaml
from scripts.calibrate_qwen import parse_args


class CalibrationConfigTests(unittest.TestCase):
    def test_registered_settings_and_explicit_overrides(self):
        required=['--dataset','unused','--frequencies','unused']
        args=parse_args(required)
        self.assertEqual((args.count,args.epsilon,args.gamma,args.device),(100000,.05,.9,'cuda'))
        args=parse_args(required+['--count','100001','--gamma','1'])
        self.assertEqual(args.resolved_config['count'],100001)
        self.assertEqual(args.resolved_config['gamma'],1.)

    def test_reject_insufficient_reference_count_and_changed_conventions(self):
        base=yaml.safe_load(Path('configs/calibration.yaml').read_text())
        for key,value in [('count',99999),('split','train'),('skip_first',0),
                          ('sequence_length',64),('exclude_final',False)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'config.yaml'
                path.write_text(yaml.safe_dump({**base,key:value}))
                with self.assertRaises(ValueError):
                    parse_args(['--dataset','unused','--frequencies','unused','--config',str(path)])
