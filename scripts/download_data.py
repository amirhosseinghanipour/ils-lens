"""Network-enabled preparation only; never called implicitly by experiments."""
import json,os
from pathlib import Path
from huggingface_hub import HfApi
from datasets import load_dataset

root=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(root/'.cache/huggingface'))
api=HfApi()
revision=api.dataset_info('Salesforce/wikitext').sha
dataset=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1',revision=revision,
                     cache_dir=str(root/'.cache/huggingface/datasets'))
destination=root/'data/wikitext-103-raw-v1'
if destination.exists():
    raise FileExistsError(f'{destination} already exists; refusing to overwrite')
dataset.save_to_disk(str(destination))
(root/'data/wikitext-manifest.json').write_text(json.dumps(dict(
    repository='Salesforce/wikitext',config='wikitext-103-raw-v1',revision=revision,
    rows={name:len(split) for name,split in dataset.items()},
    fingerprints={name:split._fingerprint for name,split in dataset.items()}),indent=2))
# Broad search is discovery evidence, not verification of architecture conventions.
queries=['Qwen2.5-0.5B lens','Qwen2.5-0.5B transcoder']
results={query:[model.id for model in api.list_models(search=query,limit=100)] for query in queries}
(root/'references/hf-artifact-search.json').write_text(json.dumps(results,indent=2))
