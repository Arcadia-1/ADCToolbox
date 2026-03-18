# ADCToolbox Testing Guide

Use the smallest test surface that covers the change.

## Test Layers

- `python/tests/unit/`
  Use for local algorithm changes, CLI logic, packaging helpers, and deterministic utilities.
- `python/tests/integration/`
  Use when the change affects real dataset processing or figure/report assembly.
- `python/tests/compare/`
  Use when MATLAB parity may have changed.

## Typical Commands

Run from `python/`:

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/compare/ -v
```

Run a narrow target first:

```bash
pytest tests/unit/dout/test_analyze_weight_radix.py -v
pytest tests/unit/calibration/test_verify_calibration_lite.py -v
pytest tests/unit/test_cap2weight.py -v
```

## Packaging And CLI Changes

- run the narrow unit test that covers the changed helper
- run a command-level smoke test against a temp output directory
- avoid touching the real `~/.codex/skills` during verification

## Public API Changes

- update or add unit tests
- update affected examples
- run integration tests only if the changed path needs them
- run compare tests only if MATLAB parity is relevant

## Skill Maintenance Changes

- verify the source files under `python/src/adctoolbox/_bundled_skills/skills/`
- run the skill installer CLI into a temp destination
- inspect the installed output tree
