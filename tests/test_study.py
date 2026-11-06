import json,tempfile,unittest
from pathlib import Path
import torch
from ilslens.study import stages,source_digest
from ilslens.graphs.compare import compare_graphs

class StudyTests(unittest.TestCase):
    def test_stage_order_guards_dependencies_and_never_requests_cpu(self):
        plan=stages('/data','/references','/corpus');names=[row['name'] for row in plan]
        self.assertEqual(len(names),len(set(names)));self.assertEqual(names[-1],'audit')
        for first,second in [('numerical','primary-collection'),('primary-collection','primary-fitting'),('readout-gates','jpp'),
            ('jpp','calibration'),('calibration','readouts'),('workspacebench','template'),('template','graphs'),('transcoders','graphs'),('planted','planted-graphs')]:
            self.assertLess(names.index(first),names.index(second))
        for row in plan:self.assertNotIn('cpu',row['command'])
    def test_source_digest_ignores_logs_but_detects_method_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'ilslens').mkdir();(root/'runs').mkdir()
            (root/'ilslens/test.py').write_text('x=1');first=source_digest(root)
            (root/'runs/log.json').write_text('{}');self.assertEqual(first,source_digest(root))
            (root/'ilslens/test.py').write_text('x=2');self.assertNotEqual(first,source_digest(root))
    def test_graph_comparison_null_and_matched_site_rejection(self):
        base=[dict(item_id=str(i),layer=0,position=p,token='owl',target_value=float(i),graph_completeness=.7,
            far_upstream_share=.2,all_edges_pass=True,graph_status='measured') for i in range(8) for p in (1,2)]
        same=compare_graphs(base,base,seed=1)
        self.assertEqual(same['summary'][0]['difference'],0);self.assertEqual(same['summary'][0]['ci'],[0,0])
        self.assertEqual(len(same['summary'][0]['control_split_half_null']),1000)
        with self.assertRaises(ValueError):compare_graphs(base,base[:-1])
