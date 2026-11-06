import unittest
import torch
from ilslens.gate import random_direction_gate
from ilslens.eval.workspacebench import phrase_gate_report


class UndefinedGateTests(unittest.TestCase):
    def test_no_direction_is_unknown_not_a_pass_or_measured_failure(self):
        def forbidden(_):raise AssertionError('Undefined intervention must not be executed')
        zero=random_direction_gate(torch.zeros(4),torch.eye(4),forbidden)
        self.assertIsNone(zero['passes']);self.assertEqual(zero['reason'],'zero_direction')
        covariance=random_direction_gate(torch.ones(4),torch.zeros(4,4),forbidden)
        self.assertEqual(covariance['reason'],'degenerate_covariance')
        rows=[dict(method='ils',family='multihop',item_id='one',gate=zero)]
        result=phrase_gate_report(rows)[0]
        self.assertIsNone(result['pass_rate']);self.assertEqual(result['undefined_interventions'],1)
        rows.append(dict(method='ils',family='multihop',item_id='two',gate=dict(passes=False)))
        result=phrase_gate_report(rows)[0]
        self.assertEqual(result['pass_rate'],0.);self.assertEqual(result['measured_interventions'],1)

    def test_nonfinite_effect_is_not_silently_a_failed_control(self):
        with self.assertRaises(FloatingPointError):
            random_direction_gate(torch.ones(4),torch.eye(4),lambda _:float('nan'))
