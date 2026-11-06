import unittest
import torch
from test_core import tiny_model
from ilslens.lora import AdapterSet,train_adapters
from ilslens.eval.subliminal import transmission_test
from ilslens.eval.budget import close_budget
from ilslens.calibration import fit_calibration

class StressTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(61)
    def test_adapter_training_preserves_base_and_disabled_output(self):
        model=tiny_model();ids=torch.randint(0,32,(1,24))
        before={name:value.detach().clone() for name,value in model.named_parameters()}
        clean=model(ids).logits.detach()
        adapters=AdapterSet(model,rank=2,alpha=4)
        torch.testing.assert_close(model(ids).logits,clean)
        labels=ids[0].clone();labels[:12]=-100
        losses=train_adapters(model,adapters,[dict(input_ids=ids[0],labels=labels)],steps=3,learning_rate=.01)
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertFalse(torch.equal(model(ids).logits,clean))
        with adapters.disabled():torch.testing.assert_close(model(ids).logits,clean)
        saved=adapters.state()
        adapters.load(saved);adapters.remove()
        for name,param in model.named_parameters():torch.testing.assert_close(param,before[name],rtol=0,atol=0)
        torch.testing.assert_close(model(ids).logits,clean)
    def test_transmission_requires_every_seed(self):
        student=[[True]*50]*3;control=[[False]*50]*3
        self.assertTrue(transmission_test(student,control)['transmitted'])
        student[2]=[False]*50
        self.assertFalse(transmission_test(student,control)['transmitted'])
    def test_unknown_budget_not_zero(self):
        report=close_budget([dict(item_id='a',lens_r2=.1,graph_completeness=None)])
        self.assertIsNone(report['fraction']);self.assertEqual(report['unknown_cases'],1)
        report=close_budget([dict(item_id='a',lens_r2=-1,graph_completeness=.1)])
        self.assertEqual(report['fraction'],1);self.assertEqual(report['rows'][0]['unexplained'],2)
    def test_calibration_refuses_small_reference_set(self):
        with self.assertRaises(ValueError):fit_calibration(torch.randn(20,4),torch.randn(8,4),torch.ones(4),torch.ones(8))
    def test_calibration_fingerprint_binds_actual_matrix(self):
        from ilslens.calibration import matrix_fingerprint
        matrix=torch.eye(4);identity=matrix_fingerprint(matrix)
        self.assertEqual(identity,matrix_fingerprint(matrix.clone()))
        matrix[0,0]+=.01
        self.assertNotEqual(identity,matrix_fingerprint(matrix))
    def test_positive_100k_calibration_keeps_dead_rows_out_of_fdr(self):
        from ilslens.calibration import fit_calibration,calibrated_readout
        h=torch.randn(100000,4);unembedding=torch.randn(8,4);gain=torch.ones(4);counts=torch.ones(8);counts[0]=0
        calibration=fit_calibration(h,unembedding,gain,counts,token_chunk=4)
        scores=unembedding@h[0]/h[0].square().mean().sqrt();scores[0]=100.
        result=calibrated_readout(scores,calibration)
        self.assertEqual(calibration['reference_count'],100000)
        self.assertNotIn(0,result['accepted_ids']);self.assertTrue(torch.isneginf(result['z'][0]))
        self.assertTrue(torch.isfinite(result['z'][1:]).all())
