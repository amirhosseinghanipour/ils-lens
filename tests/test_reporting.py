import json,tempfile,unittest
from pathlib import Path
from ilslens.reporting import cell,render,measured_tables

class ReportingTests(unittest.TestCase):
    def test_missing_stage_and_negative_r2_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'runs/faithfulness').mkdir(parents=True)
            result=dict(rows=[dict(lens='ILS',layer=0,epsilon=.05,gamma=.9,r2=-3.,ci=[-4.,-2.]),
                              dict(lens='J++ (label-tuned)',layer=0,r2=None,ci=[None,None])])
            (root/'runs/faithfulness/results.json').write_text(json.dumps(result))
            text,found,missing=render(root/'runs',root/'artifacts')
            self.assertIn('| -3 |',text);self.assertIn('unknown / undefined',text);self.assertIn('graphs',missing)
            self.assertEqual(set(found),{'faithfulness'});self.assertIn('label-tuned',text)
            exported=json.loads((root/'artifacts/faithfulness.json').read_text())
            self.assertEqual(exported['tables']['rows'][0]['r2'],-3.)
    def test_subliminal_seed_statistics_are_not_dropped(self):
        result=dict(rows=[dict(concept='owl',transmitted=False,seeds=[dict(seed_index=0,p=1.,passes=False)])])
        self.assertEqual(measured_tables('subliminal',result)['per_seed'][0]['p'],1.)
        self.assertEqual(cell('one|two\nthree'),'one\\|two three')
