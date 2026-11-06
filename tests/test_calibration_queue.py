import unittest
from scripts.calibration_after_jpp import jpp_available


class CalibrationQueueTests(unittest.TestCase):
    def test_success_and_explicit_unavailability_are_distinct(self):
        self.assertIs(jpp_available(dict(pid=1),dict(pid=1,state='jpp_completed')),True)
        self.assertIs(jpp_available(dict(pid=1),dict(pid=1,state='unavailable_no_retained_dev_items')),False)
        for state in ('failed','running_jpp','waiting_for_capability_filter',None):
            with self.subTest(state=state),self.assertRaises(RuntimeError):
                jpp_available(dict(pid=1),dict(pid=1,state=state))
        with self.assertRaises(ValueError):
            jpp_available(dict(pid=1),dict(pid=2,state='jpp_completed'))
