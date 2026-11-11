import json
from pathlib import Path
import tempfile
import unittest
from ilslens.calibration import record_calibration_timing


class CalibrationTimingTests(unittest.TestCase):
    def test_extension_preserves_original_and_incremental_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'timing.json').write_text(json.dumps(dict(seconds=120954.55)))
            record_calibration_timing(root,dict(seconds=200,fitted_maps=[dict(lens='J++',layer=0)]))
            record_calibration_timing(root,dict(seconds=10,fitted_maps=[]))
            history=json.loads((root/'timing-history.json').read_text())
            self.assertEqual([row['seconds'] for row in history],[120954.55,200,10])
            self.assertIn('not recorded',history[0]['provenance'])
            self.assertEqual(json.loads((root/'timing.json').read_text())['seconds'],10)
