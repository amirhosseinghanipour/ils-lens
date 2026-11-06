import unittest
from scripts.benchmark_after_primary import require_primary_success


class BenchmarkQueueTests(unittest.TestCase):
    def test_only_matching_completed_controller_passes(self):
        launch=dict(pid=123)
        require_primary_success(launch,dict(pid=123,state='primary_stages_completed'))
        for state in ('running','failed','waiting_for_gpu',None):
            with self.subTest(state=state):
                with self.assertRaises(RuntimeError):
                    require_primary_success(launch,dict(pid=123,state=state))
        with self.assertRaises(ValueError):
            require_primary_success(launch,dict(pid=124,state='primary_stages_completed'))
