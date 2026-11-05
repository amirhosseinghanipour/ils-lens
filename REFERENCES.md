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
