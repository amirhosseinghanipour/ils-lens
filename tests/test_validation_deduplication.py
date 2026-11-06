import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import yaml
from test_runners import script
from test_corpus_budget import IntegerTokenizer

class ValidationDeduplicationTests(unittest.TestCase):
    def test_drop_contaminated_documents_without_changing_training_bytes(self):
        module=script('deduplicate_transcoder_validation')
        rows=[]
        for i in range(13):
            row=dict(id=str(i),source='test',split='train' if i<10 else 'validation')
            text=' '.join(str(i*4+j) for j in range(4))
            if i in (4,9):row['messages']=[dict(role='user',content=text)]
            else:row['text']=text
            rows.append(row)
        rows[10]['text']=rows[0]['text']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'original.jsonl';output=root/'clean.jsonl'
            source.write_text(''.join(json.dumps(r)+'\n' for r in rows));original=source.read_bytes()
            config=root/'config.yaml';config.write_text(yaml.safe_dump(dict(token_presentations=40,chat_every=5)))
            argv=['repair','--corpus',str(source),'--output',str(output),'--config',str(config),'--length','4']
            with patch('sys.argv',argv),patch.dict('os.environ',{'ILS_MODEL_PATH':'unused'}),patch.object(module.AutoTokenizer,'from_pretrained',return_value=IntegerTokenizer()):module.main()
            self.assertEqual(source.read_bytes(),original)
            clean=[json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r for r in clean if r['split']=='train'],rows[:10])
            self.assertEqual([r['id'] for r in clean if r['split']=='validation'],['11','12'])
            manifest=json.loads(output.with_suffix('.manifest.json').read_text());audit=json.loads(output.with_suffix('.budget.json').read_text())
            self.assertTrue(manifest['training_records_unchanged']);self.assertEqual(manifest['overlapping_window_count'],1)
            self.assertTrue(audit['passes']);self.assertEqual(audit['available_selected_tokens'],40)
            self.assertEqual(audit['overlapping_validation_windows'],0)
