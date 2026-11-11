import unittest
from scripts.audit_capability_outputs import audit


class CapabilityAuditTests(unittest.TestCase):
    def test_nonzero_chat_cohort_not_described_as_raw_failure(self):
        record=dict(item=dict(task='multihop',answers=['Paris'],split='dev'),
                    correct=8,threshold=8,capability_answers=['Paris']*8+['no']*2,
                    prompt_leaks=[],output_leaks=[],retained=True,
                    config=dict(protocol='chat-adaptation-v1'))
        result=audit([record])
        self.assertEqual(result['retained'],1)
        self.assertEqual(result['splits']['dev']['retained'],1)
        self.assertEqual(result['protocols'],['chat-adaptation-v1'])
        self.assertNotIn('No items',result['inference'])
        record['threshold']=10;record['retained']=False
        result=audit([record]);self.assertEqual(result['rows'][0]['items_passing_correctness'],0)
        self.assertIn('No items',result['inference'])
        record['retained']=True
        with self.assertRaisesRegex(ValueError,'retention'):audit([record])
        record['retained']=False
        record['correct']=7
        with self.assertRaises(ValueError):audit([record])
