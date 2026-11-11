import copy
import unittest
from scripts.readouts_after_calibration import calibration_ready,validate_readouts


class ReadoutQueueTests(unittest.TestCase):
    def test_calibration_failure_is_not_baseline_unavailability(self):
        self.assertFalse(calibration_ready(dict(pid=1),dict(pid=1,state='calibration_completed',jpp_included=False)))
        for status in [dict(pid=1,state='failed'),dict(pid=2,state='calibration_completed',jpp_included=True),dict(pid=1,state='calibration_completed')]:
            with self.assertRaises((ValueError,RuntimeError)):calibration_ready(dict(pid=1),status)

    def test_missing_calibration_or_baseline_stops_completion(self):
        result=dict(complete_calibration=True,missing_calibration=[],retained_test_items=9,
                    metadata=dict(label_tuned={'logit':False,'J-reference':False,'R-reference':False,'ILS':False,'J++ (label-tuned)':True}))
        result['rows']=[dict(lens=name,task='task',layer=layer,mode=mode,k=k)
                        for name in result['metadata']['label_tuned'] for layer in list(range(24))+['any']
                        for mode in ('raw','z') for k in (1,10)]
        validate_readouts(result,True,{'task'})
        for change in ('missing','label','jpp','cohort','cell'):
            bad=copy.deepcopy(result)
            if change=='missing':bad['missing_calibration']=['J++/layer-0.pt']
            elif change=='label':bad['metadata']['label_tuned']['J++ (label-tuned)']=False
            elif change=='jpp':del bad['metadata']['label_tuned']['J++ (label-tuned)']
            elif change=='cell':bad['rows'].pop()
            else:bad['retained_test_items']=0
            with self.subTest(change=change),self.assertRaises(ValueError):validate_readouts(bad,True,{'task'})
