import unittest
from ilslens.eval.budget import close_budget


class BudgetValidationTests(unittest.TestCase):
    def test_invalid_measurement_cannot_be_hidden_by_missing_partner(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            for field in ('lens_r2', 'graph_completeness'):
                with self.subTest(value=value, field=field):
                    item=dict(item_id='a', lens_r2=None, graph_completeness=None)
                    item[field]=value
                    with self.assertRaisesRegex(ValueError, 'finite'):
                        close_budget([item])
        for value in (-.1, 1.1):
            with self.subTest(completeness=value):
                with self.assertRaises(ValueError):
                    close_budget([dict(item_id='a', lens_r2=None, graph_completeness=value)])

    def test_missing_is_unknown_but_negative_r2_is_measured(self):
        result=close_budget([
            dict(item_id='a', lens_r2=-3., graph_completeness=.1),
            dict(item_id='b', lens_r2=None, graph_completeness=.1),
            dict(item_id='c', lens_r2=.1, graph_completeness=None)])
        self.assertEqual(result['known_cases'], 1)
        self.assertEqual(result['unknown_cases'], 2)
        self.assertEqual(result['fraction'], 1.)
        self.assertEqual(result['rows'][0]['unexplained'], 4.)
        self.assertIsNone(result['rows'][1]['unexplained'])
        self.assertIsNone(result['rows'][2]['dark_matter'])


if __name__=='__main__':unittest.main()
