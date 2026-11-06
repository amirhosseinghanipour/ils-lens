# Method references

- [Jacobian Lens](https://github.com/anthropics/jacobian-lens) and
  [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html).
  Inspected revision: `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- [R-lens](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers).
- [J++ Lens](https://github.com/safety-research/jpp_lens).
  Inspected revision: `d7efc0de372e20589e899188c0cfe8434acc63ec`.
- [WorkspaceBench](https://github.com/camilablank/workspace-bench).
- [Subliminal J-lens](https://github.com/safety-research/subliminal-jlens) and
  [subliminal learning](https://github.com/MinhxLe/subliminal-learning).

Reference inspection confirmed that `jlens/hooks.py` records block outputs.
`jlens/fitting.py` uses a default maximum sequence length of 128, excludes the
first 16 and final positions, and targets the final block output before final
normalization. Its target reduction is a **sum**, followed by a source-position
mean. Its layer validator excludes the final block as a source. These details
must be preserved or explicitly reconciled in baseline comparisons.

- [FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb): the
  corpus preparation script pins the retrieved revision of `sample-10BT` before
  streaming a bounded sample. Downloaded text is not redistributed in this repo.
- [HallucinationBench](https://github.com/camila-blank/hallucination-bench),
  inspected revision `fc96bf532ce3df0461126ba0c20b1c739a3c7603`. WorkspaceBench's
  former hallucination family has moved here. Its older vendored index is not
  an evaluation of this project's Qwen checkpoint.
- [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k):
  a pinned 512-conversation sample of `train_sft` supplies chat-formatted text.
  Source documents, rather than token windows, determine the held-out partition.
- [Released exact-model J-lens](https://huggingface.co/MildHotSauce/jlens-qwen2.5-0.5b-instruct),
  revision `400efee5eda91e59327c958c006ec9ac9772b91b`; its
  [public fitting script](https://github.com/OphanimLLC/silent-thoughts/blob/844980cd460b1740973d334da3ebc58001640972/fit/fit_lens.py)
  calls unmodified `jlens.fit`. The artifact independently records 200 prompts,
  width 896 and source layers 0–22. See `configs/released-jlens.json` for the
  checksum, source/target convention evidence and missing execution provenance.
