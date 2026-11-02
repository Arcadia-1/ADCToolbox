# Foreground calibration PR validation

This PR adds four supervised foreground APIs and independent `12_foreground/`
examples. Training requires an aligned reference; normal operation evaluates a
frozen polynomial or residual MLP mapping. Learned weights are correction
parameters, not identified physical component values.

The examples reuse the existing static nonlinearity, coarse-feedback memory and
SAR models. The custom FIR converter is absent. Dither PR #102 is independent;
this PR includes no Dither models or trackers. The shared spectral ratio helper
was extracted without changing its numerical computation. A local merge preview
with the Dither branch was conflict-free.

## Validation performed

- 43 targeted tests passed; one Windows directory-symlink privilege test skipped.
- Strict Sphinx build passed (`-W --keep-going`).
- Wheel and sdist built without caches or generated output directories.
- An isolated environment installed the wheel and replayed foreground, standalone
  SAR and combined smoke experiments: 842 + 238 + 324 metric rows. Example copying
  and temporary skill installation passed.
- Extracted code replayed 4488 metric rows from the full experiments without
  refitting; the maximum independent FFT difference was 4.21e-11 dB.

These checks precede the removal of generated archives from the PR. Numerical
code and tests are unchanged by that cleanup. Detailed runs remain local;
the PR retains concise current results and scripts that generate fresh outputs.
It does not bundle historical custom-FIR results, per-record CSV dumps, machine
paths, installation inventories or large hash manifests as source assets.

Current behavioral results are in
[the packaged results summary](../../python/src/adctoolbox/examples/12_foreground/FOREGROUND_REPOSITORY_RESULTS.md).
Reference requirements, mathematical derivations and measured-hardware literature
are in the adjacent theory and workflow guides. MLP iteration-limit termination,
quantization/code-feedback limits and SAR-input calibration access remain explicit.
This is not measured-chip, transistor-level, RTL, fixed-point or throughput validation.

## Reproduce checks

From `python/`:

```powershell
uv run --extra docs --with pytest pytest tests/unit/test_foreground_workflows.py tests/unit/test_foreground_bitweights_example.py tests/unit/test_combined_foreground_example.py tests/unit/test_skill_cli.py tests/integration/test_user_guide_skill_examples.py -q -rs
uv run --extra docs sphinx-build -b html -W --keep-going docs/source <new-html-directory>
uv build --out-dir <new-dist-directory>
```

For training and frozen audits, follow the commands in `12_foreground/README.md`
and the adjacent guides, using fresh output directories. Existing `models`,
`siggen` and spectrum numerical implementations are unchanged. The one-line
`analyze_error_spectrum.py` docstring fix enables strict Sphinx and matches #102.
