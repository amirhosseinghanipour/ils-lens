import os,unittest
from pathlib import Path

class WorkspaceReferenceTests(unittest.TestCase):
    def test_registered_banks_and_actual_tokenizer_read_sites(self):
        root=Path(os.environ.get('ILS_WORKSPACE_REFERENCE','../references/workspace-bench'))
        model=os.environ.get('ILS_MODEL_PATH')
        if not root.is_dir() or not model:self.skipTest('Requires local WorkspaceBench and model tokenizer, no weights are loaded')
        from transformers import AutoTokenizer
        from ilslens.eval.workspacebench import load_family,load_reference
        tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
        plan,render,_,_,_=load_reference(root)
        for family in ['multihop','multilingual','typo','association','poetry','relational_multihop','role_bound_association','chain_intermediates']:
            rows=load_family(root,family);specs={spec.id:spec for spec in plan.plan(family,layers=[0,7,15,23])}
            self.assertTrue(rows)
            for row in rows:
                identity=row['item'].id.split(':',2)[2];spec=specs[identity]
                rendered=render(spec,tokenizer);positions=plan.resolve(spec.positions,rendered.tokens)
                self.assertTrue(positions,identity)
                self.assertTrue(all(-len(rendered.ids)<=p<len(rendered.ids) for p in positions),identity)
    def test_absence_is_not_a_hallucination_verdict(self):
        from ilslens.eval.hallucination import exact_readout_evidence
        result=exact_readout_evidence(['owl','cat'],'An owl flew away.')
        self.assertEqual(result['matched'],1);self.assertIsNone(result['hallucination_rate'])
        self.assertEqual(result['unjudged_claims'],2)
