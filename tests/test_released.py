import hashlib,tempfile,unittest
from pathlib import Path
import torch
from ilslens.released import load_released

class ReleasedLensTests(unittest.TestCase):
    def test_pinned_import_preserves_conventions_and_excludes_prior_use(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lens.pt';matrices={i:torch.randn(4,4).half() for i in (0,1)}
            torch.save(dict(J=matrices,n_prompts=200,source_layers=[0,1],d_model=4),path)
            config=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_layers=2,target_layer=2,n_prompts=200,d_model=4,stored_dtype='torch.float16')
            result=load_released(path,config);self.assertFalse(result['metadata']['prior_eligible']);self.assertTrue(result['metadata']['extra_baseline'])
            torch.testing.assert_close(result['matrices'][0],matrices[0].float());torch.testing.assert_close(result['matrices'][2],torch.eye(4))
            with self.assertRaisesRegex(ValueError,'checksum'):load_released(path,dict(config,sha256='wrong'))
            with self.assertRaisesRegex(ValueError,'target convention'):load_released(path,dict(config,target_layer=3))
