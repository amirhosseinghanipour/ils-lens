import unittest
from transformers import BatchEncoding
from ilslens.eval.contrastive import render_condition


class ChatTokenIdsTests(unittest.TestCase):
    def test_token_ids_are_explicit_when_tokenizer_defaults_to_mapping(self):
        class Tokenizer:
            def apply_chat_template(self,messages,tokenize,add_generation_prompt,return_dict=True):
                self.messages=messages
                return BatchEncoding({'input_ids':[1,2,3]}) if return_dict else [1,2,3]
        tokenizer=Tokenizer()
        self.assertEqual(render_condition(tokenizer,None,'hello'),[1,2,3])
        self.assertEqual(tokenizer.messages,[dict(role='user',content='hello')])
