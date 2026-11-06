import copy
import unittest
import torch
from ilslens.calibration import (CALIBRATION_VERSION,calibration_input_fingerprint,
                                 validate_calibration_cache)


class CalibrationResumeTests(unittest.TestCase):
    def test_changes_to_mask_references_or_transport_invalidate_cache(self):
        reference=dict(layer=0,reference_count=100000,primary_fingerprint='fixture',split='test')
        matrix=torch.eye(2);counts=torch.tensor([1,0])
        key=calibration_input_fingerprint(reference,matrix,counts)
        cache=dict(calibration_version=CALIBRATION_VERSION,metadata=dict(input_fingerprint=key),
                   reference_count=100000,records=[{},{}],dead_mask=torch.tensor([False,True]))
        validate_calibration_cache(cache,key,100000,2)
        for changed in [calibration_input_fingerprint(reference,matrix,torch.tensor([1,1])),
                        calibration_input_fingerprint({**reference,'layer':1},matrix,counts),
                        calibration_input_fingerprint(reference,matrix*2,counts)]:
            with self.assertRaises(ValueError):validate_calibration_cache(cache,changed,100000,2)
        for field,value in [('reference_count',99999),('records',[{}]),
                            ('dead_mask',torch.zeros(3)),('metadata',{})]:
            bad=copy.deepcopy(cache);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_calibration_cache(bad,key,100000,2)
