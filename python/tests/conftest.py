from pathlib import Path
import matplotlib
import pytest

matplotlib.use("Agg")

@pytest.fixture
def project_root():
    """Fixture to provide the project root directory."""    
    return Path(__file__).resolve().parents[2]
