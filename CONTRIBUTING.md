# Contributing

Keep research claims tied to recorded experiments. Report negative findings and
failed validation checks. Do not choose hyperparameters using evaluation labels;
label-tuned variants must be identified separately. Record method changes with
their reasons and preserve enough provenance to reproduce comparisons.

Before submitting changes:

```sh
python -m pip install -e '.[test]'
sh scripts/test.sh
```

Add tests when changing numerical behavior, intervention semantics, or data
splits. Include seeds, data revisions, model conventions and compute estimates
for experiments. Run tiny inputs before launching expensive jobs. Never commit
credentials, model downloads, private datasets, environments or machine-specific
paths. Keep large generated datasets outside Git.

Use focused commits with a clear explanation of the change and validation.
Bug reports should include the command, package versions, minimal input, and
expected versus observed behavior. Research comparisons need uncertainty estimates
and matched controls before causal or improvement claims.
