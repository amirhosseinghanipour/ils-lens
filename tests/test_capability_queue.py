import unittest
from scripts.capability_after_benchmark import require_benchmark_success


class CapabilityQueueTests(unittest.TestCase):
    def test_only_matching_successful_benchmark_unlocks_filter(self):
        launch=dict(pid=123)
        require_benchmark_success(launch,dict(pid=123,state='benchmark_completed'))
        for state in ('running_benchmark','failed','waiting_for_primary',None):
            with self.subTest(state=state),self.assertRaises(RuntimeError):
                require_benchmark_success(launch,dict(pid=123,state=state))
        with self.assertRaises(ValueError):
            require_benchmark_success(launch,dict(pid=124,state='benchmark_completed'))
