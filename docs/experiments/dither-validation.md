# Dither validation

Checks for the Dither examples, the public `DitherGainTracker`,
`apply_dither_gain` and `DitherHDCTracker` APIs, and the algebra behind the
[model basis](hdc-model-basis.md) and the
[switch-phase derivation](hdc-physical-state-and-phases.md). The long
experiments are summarized in [dither-results.md](dither-results.md) and are
not repeated here.

## Commands

From `python/`, with fresh output directories:

```powershell
uv run --with pytest pytest tests/unit tests/integration/test_user_guide_skill_examples.py -q
uv run --extra docs --extra accel --with pytest pytest tests/unit/test_dither_api.py tests/unit/test_dither_background_example.py tests/unit/test_dither_mechanisms.py tests/unit/test_hdc_fixed.py tests/unit/test_hdc_small_signal.py tests/unit/test_hdc_correlation.py tests/unit/test_joint_dither.py tests/unit/test_joint_dither_convergence.py tests/unit/test_pn_architectures.py tests/unit/test_skill_cli.py tests/integration/test_user_guide_skill_examples.py -q
uv run --no-sync --extra docs sphinx-build -b html -W --keep-going docs/source <new-docs-output>
uv run --no-sync --extra docs sphinx-build -b html -W --keep-going -D language=zh_CN docs/source <new-zh-docs-output>
uv build --out-dir <new-dist-output>
```

Reduced runs check that each experiment entry point completes and writes its
outputs. They are far too short to reproduce the archived results:

```powershell
uv run --with numba python src/adctoolbox/examples/11_calibration/exp_cal01_dither_background.py --blocks 64 --block-size 2048 --foreground-blocks 2 --seeds 11 --scenarios constant step --waveforms sine --output <new-dir>
uv run --with numba python src/adctoolbox/examples/11_calibration/exp_cal02_dither_coverage.py --blocks 16 --block-size 2048 --foreground-blocks 1 --seeds 11 --waveforms sine --scenarios constant step --output <new-dir>
uv run --with numba python src/adctoolbox/examples/11_calibration/exp_cal06_joint_dither.py --blocks 16 --block-size 2048 --seeds 11 --waveforms sine --scenarios constant step --output <new-dir>
uv run --with numba python src/adctoolbox/examples/11_calibration/exp_cal07_joint_convergence.py --blocks 8 --long-blocks 16 --block-size 1024 --seeds 101 --output <new-dir>
uv run --with numba python src/adctoolbox/examples/11_calibration/exp_cal08_hdc_fixed.py --total-log2 22 --window-log2 18 --seeds 101 --kinds sine --output <new-dir>
uv run --with numba python src/adctoolbox/examples/11_calibration/exp_cal09_pn_architectures.py --quick --seeds 707 --output <new-dir>
uv run --with numba python src/adctoolbox/examples/11_calibration/hdc_small_signal.py --seeds 111 --noise-rms 0.0002 --train-log2 20 --test-log2 20 --window-log2 18 --short-log2 16 --output <new-dir>
uv run --with numba python src/adctoolbox/examples/11_calibration/hdc_fixed_controls.py --output <new-dir>
uv run --with numba python src/adctoolbox/examples/11_calibration/hdc_correlation.py --quick --seeds 601 --checkpoints 16 --noise-rms 0.0002 --output <new-dir>
```

From the repository root:

```powershell
python docs/experiments/check_hdc_model_basis_20261102.py --output <new-model-audit.json>
python docs/experiments/check_hdc_physical_equations_20261102.py --model-audit <new-model-audit.json> --output <new-physical-audit.json>
```

Run these two commands in order on the same checkout. The physical-equations
check validates the source hashes in the newly generated model-basis audit,
including the shared `calibration/_dither_hdc.py` kernel. The committed JSON
files remain historical records; their byte hashes can differ because of
source edits or platform line endings. Neither audit is circuit validation.

## Current results

With Python 3.13.1 on Windows:

- `tests/unit` and the skill example tests without Numba: 1047 passed,
  1 skipped and 1 xfailed; the default selection deselects 11 slow tests.
- The targeted command with the `accel` extra: 153 passed, 1 skipped. The
  skipped test needs permission to create directory symlinks.
- Strict English and zh_CN Sphinx builds and wheel/sdist builds passed; the
  wheel declares the `accel` extra.
- All nine reduced runs completed without errors.
- The model-basis check reproduces the 216 exact charge/inversion identities in
  the [stored record](hdc-model-basis-audit-20261102.json). Fresh records track
  the current source bytes and also include the extracted HDC kernel.
- The original physical-equations command required restoring historical source
  versions to match the committed model-basis hashes. The current command
  instead takes a fresh `--model-audit` as shown above and reproduces the
  [stored record](hdc-physical-state-phases-audit-20261102.json) apart from the
  source and audit provenance hashes: 512 state transitions in ideal virtual
  ground (four reset conditions), 1536 finite-gain state transitions, 384 static finite-gain
  charge relations and 128 equal-residue cancellations.
