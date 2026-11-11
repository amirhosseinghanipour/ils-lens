import unittest
from types import SimpleNamespace
import torch
from ilslens.eval.prompting import render_read_prompt,read_site
from ilslens.eval.readout_tasks import ReadoutItem,gate_item


class CharacterTokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return ''.join(f"<{m['role']}>\n{m['content']}\n" for m in messages)+'<assistant>\n'
    def __call__(self,text,return_offsets_mapping=False,**kwargs):
        if return_offsets_mapping:return dict(offset_mapping=[(i,i+1) for i in range(len(text))])
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) for c in text]]))


class ReadoutPromptingTests(unittest.TestCase):
    def test_raw_stays_identical_and_unknown_formats_fail(self):
        tokenizer=CharacterTokenizer()
        self.assertEqual(render_read_prompt(tokenizer,'hello'),'hello')
        for fmt,system in [('unknown',None),('raw','extra')]:
            with self.assertRaises(ValueError):render_read_prompt(tokenizer,'hello',fmt,system)

    def test_poetry_reads_content_newline_not_chat_delimiter(self):
        tokenizer=CharacterTokenizer();prompt='one line\nnext line'
        rendered=render_read_prompt(tokenizer,prompt,'chat','Short answer only.')
        ids,position=read_site(tokenizer,prompt,'poetry','chat','Short answer only.')
        self.assertEqual(position,rendered.index('one line')+len('one line'))
        self.assertEqual(int(ids[0,position]),ord('\n'))
        _,ordinary=read_site(tokenizer,prompt,'multihop','chat','Short answer only.')
        self.assertEqual(ordinary,len(rendered)-1)
        self.assertLess(position,ordinary)

    def test_system_context_is_included_in_leakage_gate(self):
        item=ReadoutItem('fixture','multihop','How many legs?','How many legs?',['4'],['cat'],'test')
        record=gate_item(item,['4']*10,['4']*10,rendered_prompt='system: cat\nHow many legs?')
        self.assertEqual(record['correct'],10)
        self.assertFalse(record['retained'])
        self.assertEqual(record['prompt_leaks'],['cat'])
