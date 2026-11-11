"""Explicit raw/chat prompt protocols shared by gates, read sites and dev fitting."""

def render_read_prompt(tokenizer,prompt,prompt_format='raw',system_prompt=None):
    if prompt_format=='raw':
        if system_prompt is not None:raise ValueError('Raw protocol cannot silently add a system prompt')
        return prompt
    if prompt_format!='chat':raise ValueError('Unknown readout prompt format')
    messages=([dict(role='system',content=system_prompt)] if system_prompt else [])+[dict(role='user',content=prompt)]
    return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)


def read_site(tokenizer,prompt,task,prompt_format='raw',system_prompt=None):
    rendered=render_read_prompt(tokenizer,prompt,prompt_format,system_prompt)
    tokens=tokenizer(rendered,return_tensors='pt',add_special_tokens=False).input_ids
    position=tokens.shape[1]-1
    if task.split(':')[0]=='poetry':
        newline=prompt.rfind('\n');start=rendered.rfind(prompt)
        if newline<0 or start<0:raise ValueError('Cannot locate poetry content newline in rendered prompt')
        encoded=tokenizer(rendered,return_offsets_mapping=True,add_special_tokens=False)
        candidates=[j for j,(a,b) in enumerate(encoded['offset_mapping']) if a<=start+newline<b]
        if not candidates:raise ValueError('Cannot identify poetry newline token')
        position=candidates[-1]
    return tokens,position
