from pathlib import Path
import matplotlib
import pytest

matplotlib.use("Agg")


def _resolve_comparison_output_root(root):
    if root.name != "test_output" and (root / "test_output").is_dir():
        return root / "test_output"

    return root


def pytest_addoption(parser):
    parser.addoption(
        "--artifact-root",
        action="store",
        default=None,
        help=(
            "Directory for generated integration artifacts. Defaults to a "
            "pytest tmp_path so normal test runs do not write into the repo."
        ),
    )
    parser.addoption(
        "--comparison-output-root",
        action="store",
        default=None,
        help=(
            "Directory containing generated *_python.csv comparison outputs. "
            "May be either an artifact root or its test_output subdirectory. "
            "Compare tests skip without this explicit golden-workflow input."
        ),
    )


@pytest.fixture
def project_root():
    """Fixture to provide the project root directory."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def artifact_root(request, project_root, tmp_path):
    """Root for generated integration artifacts."""
    configured_root = request.config.getoption("--artifact-root")
    if configured_root:
        root = Path(configured_root)
        return root if root.is_absolute() else project_root / root

    return tmp_path / "artifacts"


@pytest.fixture
def comparison_output_root(request, project_root):
    """Explicit Python output root for MATLAB/Python comparison tests."""
    configured_root = request.config.getoption("--comparison-output-root")
    if not configured_root:
        pytest.skip(
            "comparison tests require --comparison-output-root=<artifact-root> "
            "or --comparison-output-root=<artifact-root>/test_output; "
            "run integration tests with --artifact-root=<artifact-root> first"
        )

    root = Path(configured_root)
    root = root if root.is_absolute() else project_root / root
    return _resolve_comparison_output_root(root)
