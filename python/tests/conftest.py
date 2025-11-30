from pathlib import Path
import pytest

@pytest.fixture
def project_root():
    """Fixture to provide the project root directory."""    
    return Path(__file__).resolve().parents[2]
