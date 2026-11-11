import copy
import unittest
from scripts.faithfulness_after_jpp import validate_grid


class FaithfulnessGridTests(unittest.TestCase):
    def fixture(self, jpp=True):
        config=dict(layers=[0,1],epsilons=[.01,.1],gammas=[0.,1.],prompts=4,samples_per_prompt=2)
        names=['logit','J-reference','R-reference','ILS']+(['J++ (label-tuned)'] if jpp else [])
        rows=[dict(lens=name,layer=l,epsilon=e,gamma=g,N=8,label_tuned=name=='J++ (label-tuned)')
              for name in names for l in config['layers'] for e in config['epsilons'] for g in config['gammas']]
        return dict(rows=rows),config

    def test_no_silent_baseline_or_sample_loss(self):
        data,config=self.fixture()
        self.assertEqual(len(validate_grid(data,config,True)),5)
        for change in ('omit_cell','duplicate','omit_jpp','omit_ils','sample_count','label'):
            bad=copy.deepcopy(data)
            if change=='omit_cell':bad['rows'].pop()
            elif change=='duplicate':bad['rows'].append(bad['rows'][0])
            elif change in ('omit_jpp','omit_ils'):
                name='J++ (label-tuned)' if change=='omit_jpp' else 'ILS'
                bad['rows']=[r for r in bad['rows'] if r['lens']!=name]
            elif change=='sample_count':bad['rows'][0]['N']=7
            else:bad['rows'][0]['label_tuned']=True
            with self.subTest(change=change),self.assertRaises(ValueError):validate_grid(bad,config,True)
        data,config=self.fixture(False)
        self.assertEqual(len(validate_grid(data,config,False)),4)
        with self.assertRaises(ValueError):validate_grid(data,config,True)
