import unittest
import torch
from ilslens.eval.diagnostics import procrustes_experts

class PolarInterpretationTests(unittest.TestCase):
    def test_alignment_does_not_remove_scale_disagreement(self):
        maps=torch.stack([torch.eye(4)*scale for scale in (1.,2.,3.,4.)])
        result=procrustes_experts(maps);d=result['polar_diagnostics']
        torch.testing.assert_close(result['aligned'],result['plain'])
        self.assertAlmostEqual(d['raw_dispersion'],5.)
        self.assertAlmostEqual(d['aligned_dispersion'],5.)
        self.assertAlmostEqual(d['positive_factor_dispersion'],5.)
        self.assertEqual(d['rotation_chordal_dispersion'],0.)
    def test_rotated_positive_factors_preserve_scale_variance(self):
        generator=torch.Generator().manual_seed(919)
        rotation=torch.linalg.qr(torch.randn(8,4,4,generator=generator)).Q
        scales=torch.diag_embed(torch.rand(8,4,generator=generator)+.1)
        result=procrustes_experts(rotation@scales);d=result['polar_diagnostics']
        self.assertAlmostEqual(d['aligned_dispersion'],d['positive_factor_dispersion'],places=5)
        self.assertGreater(d['rotation_chordal_dispersion'],0.)
    def test_invalid_experts_rejected(self):
        for maps in (torch.zeros(0,4,4),torch.zeros(8,4,3),torch.full((8,4,4),float('nan'))):
            with self.assertRaises(ValueError):procrustes_experts(maps)
