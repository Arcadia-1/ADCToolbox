# ADCToolbox Test Suite

Run tests from the `python/` directory:

```bash
cd python
uv run --with pytest pytest tests -q
```

## Layout

| Directory | Purpose |
|---|---|
| `tests/unit/` | Fast unit and regression tests for individual modules |
| `tests/integration/` | Workflow tests that exercise packaged APIs and generated outputs |
| `tests/compare/` | MATLAB/Python comparison helpers and golden-reference checks |

## Common Commands

```bash
# Unit tests only
uv run --with pytest pytest tests/unit -q

# Integration tests only
uv run --with pytest pytest tests/integration -q

# Comparison tests only
uv run --with pytest pytest tests/compare -q

# Smoke-test bundled Codex skill examples
uv run --with pytest pytest tests/integration/test_user_guide_skill_examples.py -q

# Validate bundled skill installer behavior
uv run --with pytest pytest tests/unit/test_skill_cli.py -q
```

Several integration and comparison tests write regenerated CSV/PNG artifacts to
ignored output folders such as `test_output/`, `test_plots/`, and
`test_comparison_logs/`.

## MATLAB Comparison Flow

Some comparison tests expect Python outputs to exist before comparing them with
MATLAB reference CSV files:

```bash
uv run --with pytest pytest tests/integration/test_sine_fit.py -q
uv run --with pytest pytest tests/compare/test_compare_sine_fit.py -q
```

For a consolidated comparison report:

```bash
uv run python -m tests.compare.run_all_comparisons
```

## Notes

- Prefer adding narrow unit tests for new behavior, then integration tests when
  a workflow, example, or public API contract changes.
- Keep generated outputs out of git; they are covered by `.gitignore`.
