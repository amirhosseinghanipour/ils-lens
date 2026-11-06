import copy,unittest
from ilslens.validation_gate import validate_numerical_result


class ValidationGateTests(unittest.TestCase):
    def test_requires_complete_matching_passing_measurements(self):
        manifest=dict(fingerprint='test',protocol=dict(config=dict(layers=list(range(24)),epsilons=[.01,.0001])))
        result=dict(fingerprint='test',smallest_epsilon_passes_all_layers=True,
                    rows=[dict(layer=l,epsilon=e,matrix_cosine=.99,heldout_top10_overlap=.9)
                          for l in range(24) for e in [.01,.0001]])
        self.assertEqual(validate_numerical_result(result,manifest)['cells'],48)
        for mutate in [lambda r:r['rows'].pop(),lambda r:r['rows'].append(r['rows'][0]),
                       lambda r:r.update(fingerprint='wrong'),lambda r:r['rows'][1].update(matrix_cosine=.94),
                       lambda r:r['rows'][1].update(heldout_top10_overlap=.79),
                       lambda r:r['rows'][1].update(matrix_cosine=float('nan'))]:
            bad=copy.deepcopy(result);mutate(bad)
            with self.assertRaises(ValueError):validate_numerical_result(bad,manifest)
