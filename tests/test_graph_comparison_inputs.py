import copy,unittest
from ilslens.graphs.compare import compare_graphs

class GraphComparisonInputsTests(unittest.TestCase):
    def rows(self):
        return [dict(item_id=str(i),layer=0,position=2,token='owl',target_value=float(i),graph_completeness=.7,
            far_upstream_share=.2,all_edges_pass=True,graph_status='measured',
            context_fingerprint=f'context-{i}',target_fingerprint='target') for i in range(4)]
    def test_duplicate_sites_are_rejected(self):
        rows=self.rows()
        with self.assertRaisesRegex(ValueError,'Duplicate'):compare_graphs(rows+[rows[0]],rows)
    def test_matched_ids_do_not_override_changed_or_missing_inputs(self):
        for field in ('context_fingerprint','target_fingerprint'):
            for missing in (False,True):
                with self.subTest(field=field,missing=missing):
                    base=self.rows();tuned=copy.deepcopy(base)
                    if missing:del tuned[0][field]
                    else:tuned[0][field]='changed'
                    with self.assertRaises(ValueError):compare_graphs(base,tuned)
    def test_unmeasured_pairs_are_preserved(self):
        base=self.rows();tuned=copy.deepcopy(base)
        tuned[0].update(graph_status='unsupported_target',graph_completeness=None,all_edges_pass=None)
        result=compare_graphs(base,tuned)
        self.assertEqual(len(result['rows']),4)
        self.assertEqual(result['measured_pairs'],3);self.assertEqual(result['unmeasured_pairs'],1)
        row=next(r for r in result['rows'] if r['item_id']=='0')
        self.assertEqual(row['status'],'unmeasured_pair');self.assertIsNone(row['difference'])
        self.assertEqual(row['tuned_graph_status'],'unsupported_target')
        self.assertEqual(result['summary'][0]['status'],'insufficient_prompts_for_null')
    def test_nonfinite_measured_targets_are_errors(self):
        base=self.rows();tuned=copy.deepcopy(base);tuned[0]['target_value']=float('nan')
        with self.assertRaises(ValueError):compare_graphs(base,tuned)
