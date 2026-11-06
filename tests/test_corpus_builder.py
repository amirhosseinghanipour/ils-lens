import json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import yaml
from test_runners import script


class CorpusBuilderTests(unittest.TestCase):
    def test_stage_order_and_failure_stop(self):
        module=script('build_transcoder_corpus')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'corpus.jsonl';config=root/'config.yaml'
            config.write_text(yaml.safe_dump(dict(web_documents=200000,chat_documents=20000,
                web_revision='pinned-web',chat_revision='pinned-chat')))
            argv=['build','--shards',str(root/'shards'),'--output',str(output),'--config',str(config)]
            with patch('sys.argv',argv),patch.object(module.subprocess,'run') as run:
                module.main()
            calls=[call.args[0] for call in run.call_args_list]
            self.assertEqual([call[1] for call in calls],['scripts/download_transcoder_shards.py',
                'scripts/prepare_transcoder_corpus.py','scripts/audit_transcoder_corpus.py'])
            self.assertIn('pinned-web',calls[1]);self.assertIn('pinned-chat',calls[1])
            self.assertEqual(json.loads(output.with_suffix('.build-status.json').read_text())['state'],'completed')
            with patch('sys.argv',argv),patch.object(module.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['download'])) as run:
                with self.assertRaises(subprocess.CalledProcessError):module.main()
            self.assertEqual(run.call_count,1)
            self.assertEqual(json.loads(output.with_suffix('.build-status.json').read_text())['state'],'failed')
