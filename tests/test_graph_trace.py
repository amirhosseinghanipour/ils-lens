import json,unittest
import torch
from test_core import tiny_model
from ilslens.transcoders.train import SkipTranscoder
from ilslens.graphs.qwen import QwenGraph
from ilslens.graphs.trace import trace_graph

class FeatureGraphTests(unittest.TestCase):
    def test_feature_parents_form_validated_dag_and_lens_is_only_a_target(self):
        torch.set_num_threads(2);model=tiny_model();ids=torch.randint(0,32,(1,24))
        transcoders=[SkipTranscoder(16).eval().requires_grad_(False) for _ in range(4)]
        graph=QwenGraph(model,ids,transcoders)
        result=trace_graph(graph,dict(kind='lens',layer=3,position=20,vector=torch.randn(16)),
            max_feature_targets=4,max_depth=2,parents_per_target=2,edges_per_kind=2)
        self.assertTrue(result['all_checked_edges_pass']);self.assertTrue(result['expanded_features'])
        self.assertEqual(result['targets'][0]['kind'],'lens')
        nodes={node['id']:node for node in result['nodes']}
        self.assertTrue(all(node['kind'] in ('feature','error','embedding') for node in nodes.values()))
        for edge in result['edges']:
            if edge['destination_is_target']:continue
            source=nodes[edge['source']];destination=nodes[edge['destination']]
            if source['layer'] is not None:self.assertLess(source['layer'],destination['layer'])
            self.assertLessEqual(source['position'],destination['position'])
        json.dumps(result,allow_nan=False)
