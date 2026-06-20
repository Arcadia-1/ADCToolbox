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

Integration tests generate CSV/PNG artifacts under a pytest temporary directory
by default. To keep outputs for MATLAB/Python comparison work, pass an explicit
artifact root:

```bash
uv run --with pytest pytest tests/integration/test_basic.py -q --artifact-root ../tmp-artifacts
```

## MATLAB Comparison Flow

Comparison tests do not read repository-level `test_output/` by default. They
skip unless a generated output root is passed explicitly:

```bash
uv run --with pytest pytest tests/integration/test_basic.py -q --artifact-root ../tmp-artifacts
uv run --with pytest pytest tests/compare/test_compare_basic.py -q --comparison-output-root ../tmp-artifacts
```

For a consolidated comparison report:

```bash
uv run python -m tests.compare.run_all_comparisons
```

Some MATLAB golden datasets are not committed yet. In those cases the strict
comparison runner reports the missing reference explicitly.

## MATLAB Test Runner

The MATLAB test suite is optional and requires a local MATLAB installation. From
the repository root:

```bash
python matlab/tests/run_matlab_tests.py all
```

If MATLAB is optional in your environment:

```bash
python matlab/tests/run_matlab_tests.py all --missing-ok
```

See `matlab/MATLAB_TESTS_OVERVIEW.md` for suites, environment variables, and
exit-code behavior.

## Notes

- Prefer adding narrow unit tests for new behavior, then integration tests when
  a workflow, example, or public API contract changes.
- Keep generated outputs out of git; they are covered by `.gitignore`.
