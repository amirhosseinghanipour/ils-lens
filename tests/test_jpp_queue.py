import unittest
from scripts.jpp_after_capability import require_capability_success,retained_dev_items


class JppQueueTests(unittest.TestCase):
    def test_matching_completed_predecessor_required(self):
        require_capability_success(dict(pid=123),dict(pid=123,state='capability_filter_completed'))
        for state in ('running_capability_filter','waiting_for_benchmark','failed',None):
            with self.subTest(state=state),self.assertRaises(RuntimeError):
                require_capability_success(dict(pid=123),dict(pid=123,state=state))
        with self.assertRaises(ValueError):
            require_capability_success(dict(pid=123),dict(pid=124,state='capability_filter_completed'))

    def test_test_labels_and_failed_capability_items_do_not_unlock_fitting(self):
        gates=[dict(retained=True,item=dict(id='test',split='test')),
               dict(retained=False,item=dict(id='failed-dev',split='dev'))]
        self.assertEqual(retained_dev_items(gates),[])
        gates.append(dict(retained=True,item=dict(id='passed-dev',split='dev')))
        self.assertEqual(retained_dev_items(gates),[dict(id='passed-dev',split='dev')])
