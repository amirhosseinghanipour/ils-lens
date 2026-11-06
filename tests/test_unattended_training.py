import copy
import unittest
from scripts.unattended_training import require_calibration_success,validate_benchmark


class UnattendedTrainingTests(unittest.TestCase):
    def test_only_matching_success_unlocks_training(self):
        require_calibration_success(dict(pid=1),dict(pid=1,state='calibration_completed'))
        for state in ('running_calibration','failed','waiting_for_jpp',None):
            with self.subTest(state=state),self.assertRaises(RuntimeError):
                require_calibration_success(dict(pid=1),dict(pid=1,state=state))
        with self.assertRaises(ValueError):
            require_calibration_success(dict(pid=1),dict(pid=2,state='calibration_completed'))

    def test_benchmark_must_cover_unchanged_registered_budget(self):
        config=dict(unique_training_windows=True)
        manifest=dict(protocol=dict(training_config=config))
        result=dict(registered_token_budget=100000000,layers=24,dtype='float32',tf32=False)
        validate_benchmark(config,manifest,result)
        for key,value in [('registered_token_budget',1000),('layers',4),('dtype','bfloat16'),('tf32',True)]:
            changed=copy.deepcopy(result);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_benchmark(config,manifest,changed)
        with self.assertRaises(ValueError):validate_benchmark(dict(unique_training_windows=False),manifest,result)
