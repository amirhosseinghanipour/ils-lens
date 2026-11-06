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
            ('jpp','calibration'),('calibration','readouts'),('workspacebench','graph-items'),('graph-items','template'),('template','graphs'),('transcoder-benchmark','transcoders'),('transcoders','graphs'),('planted','planted-graphs')]:
            self.assertLess(names.index(first),names.index(second))
        template=next(row for row in plan if row['name']=='template')
        self.assertIn('--dictionary-items',template['command'])
        for row in plan:self.assertNotIn('cpu',row['command'])
    def test_source_digest_ignores_logs_but_detects_method_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'ilslens').mkdir();(root/'runs').mkdir()
            (root/'ilslens/test.py').write_text('x=1');first=source_digest(root)
            (root/'runs/log.json').write_text('{}');self.assertEqual(first,source_digest(root))
            (root/'ilslens/test.py').write_text('x=2');self.assertNotEqual(first,source_digest(root))
    def test_graph_comparison_null_and_matched_site_rejection(self):
        base=[dict(item_id=str(i),layer=0,position=p,token='owl',target_value=float(i),graph_completeness=.7,
            far_upstream_share=.2,all_edges_pass=True,graph_status='measured',context_fingerprint=f'context-{i}',target_fingerprint='target') for i in range(8) for p in (1,2)]
        same=compare_graphs(base,base,seed=1)
        self.assertEqual(same['summary'][0]['difference'],0);self.assertEqual(same['summary'][0]['ci'],[0,0])
        self.assertEqual(len(same['summary'][0]['control_split_half_null']),1000)
        with self.assertRaises(ValueError):compare_graphs(base,base[:-1])

    def test_explicit_primary_directory_routes_all_consumers_and_audit(self):
        plan=stages('/data','/references','/corpus',primary_run='/registered/primary-v4')
        routed=[]
        for stage in plan:
            command=stage['command']
            if '--run' in command or '--primary-run' in command or stage['name'] in ('gpu-benchmark','primary-collection'):
                option='--run' if '--run' in command else '--primary-run' if '--primary-run' in command else '--output'
                self.assertEqual(command[command.index(option)+1],'/registered/primary-v4')
                routed.append(stage['name'])
            self.assertNotIn('runs/qwen-main',command)
        self.assertEqual(len(routed),22)
        from test_runners import script
        module=script('audit_study')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            # A malformed legacy manifest must not be read for an explicit run.
            (root/'qwen-main').mkdir();(root/'qwen-main/manifest.json').write_text('invalid JSON')
            result=module.audit(root,root/'registered-primary')
            self.assertFalse(result['complete'])
            self.assertFalse(next(c for c in result['checks'] if c['criterion']=='primary_manifest')['passes'])
