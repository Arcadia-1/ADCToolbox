"""The documented HDC audit pipeline must work on a fresh checkout."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
AUDITS = ROOT / "docs" / "experiments"
KERNEL = "python/src/adctoolbox/calibration/_dither_hdc.py"


def run_audit(script, *args):
    return subprocess.run(
        [sys.executable, str(AUDITS / script), *map(str, args)],
        capture_output=True, text=True,
    )


@pytest.fixture
def model_audit(tmp_path):
    path = tmp_path / "model.json"
    result = run_audit("check_hdc_model_basis_20261102.py", "--output", path)
    assert result.returncode == 0, result.stderr
    return path


def test_current_model_audit_feeds_physical_equations(model_audit, tmp_path):
    basis = json.loads(model_audit.read_text())
    assert basis["exact_charge_identity_cases"] == 216
    assert KERNEL in basis["source_sha256"]
    output = tmp_path / "physical.json"
    result = run_audit("check_hdc_physical_equations_20261102.py",
                       "--model-audit", model_audit, "--output", output)
    assert result.returncode == 0, result.stderr
    physical = json.loads(output.read_text())
    assert sum(physical["state_charge_identity_samples"].values()) == 512
    assert physical["finite_open_loop_gain_state_identity_samples"] == 1536
    assert physical["finite_open_loop_gain_charge_identity_cases"] == 384
    assert physical["equal_reset_PN_and_DAC_tail_cancellation_samples"] == 128
    assert physical["source_sha256"] == basis["source_sha256"]


def test_physical_audit_rejects_stale_model_sources(model_audit, tmp_path):
    basis = json.loads(model_audit.read_text())
    basis["source_sha256"][KERNEL] = "0" * 64
    model_audit.write_text(json.dumps(basis))
    output = tmp_path / "physical.json"
    result = run_audit("check_hdc_physical_equations_20261102.py",
                       "--model-audit", model_audit, "--output", output)
    assert result.returncode != 0
    assert KERNEL in result.stderr
    assert not output.exists()
