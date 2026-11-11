import hashlib
import unittest
from ilslens.eval.readout_tasks import validate_labeled_dev,normalize


class DevProvenanceTests(unittest.TestCase):
    def test_renamed_prompt_and_changed_gate_rejected(self):
        gates=[dict(item=dict(id='renamed',prompt='  SAME question! ',split='test'))]
        meta=dict(dev_gate_sha256='gate',dev_item_ids=['other'],
                  dev_prompt_hashes=[hashlib.sha256(normalize('same question').encode()).hexdigest()])
        with self.assertRaisesRegex(ValueError,'normalized prompt'):validate_labeled_dev(gates,meta,'gate')
        meta['dev_prompt_hashes']=[]
        validate_labeled_dev(gates,meta,'gate')
        with self.assertRaisesRegex(ValueError,'dev gates differ'):validate_labeled_dev(gates,meta,'different')
        meta['dev_item_ids']=['renamed']
        with self.assertRaisesRegex(ValueError,'item ID'):validate_labeled_dev(gates,meta,'gate')
        del meta['dev_prompt_hashes']
        with self.assertRaisesRegex(ValueError,'lacks prompt hashes'):validate_labeled_dev(gates,meta,'gate')
