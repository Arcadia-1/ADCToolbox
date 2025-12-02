"""
Batch runner for all digital output comparison tests.

This module runs all comparison tests for digital output (dout) tools:
- bit_activity
- enob_bit_sweep
- fg_cal_sine
- weight_scaling
- fg_cal_sine_overflow_chk

Usage:
    pytest run_digital_comparisons.py -v
"""

import pytest
from pathlib import Path

from tests.compare.test_compare_bit_activity import test_compare_bit_activity
from tests.compare.test_compare_enob_bit_sweep import test_compare_enob_bit_sweep
from tests.compare.test_compare_fg_cal_sine import test_compare_fg_cal_sine
from tests.compare.test_compare_weight_scaling import test_compare_weight_scaling
from tests.compare.test_compare_fg_cal_sine_overflow_chk import test_compare_fg_cal_sine_overflow_chk


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.parent


def test_digital_bit_activity(project_root):
    """Test bit_activity comparison."""
    test_compare_bit_activity(project_root)


def test_digital_enob_bit_sweep(project_root):
    """Test enob_bit_sweep comparison."""
    test_compare_enob_bit_sweep(project_root)


def test_digital_fg_cal_sine(project_root):
    """Test fg_cal_sine comparison."""
    test_compare_fg_cal_sine(project_root)


def test_digital_weight_scaling(project_root):
    """Test weight_scaling comparison."""
    test_compare_weight_scaling(project_root)


def test_digital_fg_cal_sine_overflow_chk(project_root):
    """Test fg_cal_sine_overflow_chk comparison."""
    test_compare_fg_cal_sine_overflow_chk(project_root)
