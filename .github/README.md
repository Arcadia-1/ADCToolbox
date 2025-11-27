# GitHub CI/CD Configuration

## Workflows

### `ci.yml` - Unit Tests
Runs automatically on:
- Pull requests to `main` branch
- Pushes to `main` branch

**What it does:**
- Runs all MATLAB unit tests in `matlab/tests/unit/`
- Runs all Python unit tests in `python/tests/unit/`
- Generates code coverage reports
- Archives test results (kept for 7 days)

**Test results:**
- ✅ Green check = All tests passed
- ❌ Red X = Some tests failed (click "Details" to see which ones)

## Viewing Test Results

1. Go to the "Actions" tab in GitHub
2. Click on the latest workflow run
3. Click on "matlab-unit-tests" or "python-tests" to see details
4. Download artifacts (test results, coverage reports) if needed

## Local Testing

Before pushing, you can run tests locally:

**MATLAB:**
```matlab
cd matlab/tests/unit
% Run individual test
test_sineFit
```

**Python:**
```bash
cd python/tests/unit
pytest -v
```

## Future Enhancements

Planned additions:
- System tests on `main` branch only (slower, full integration)
- Automatic artifact archiving (figures, datasets)
- Release automation on version tags
- Documentation generation
