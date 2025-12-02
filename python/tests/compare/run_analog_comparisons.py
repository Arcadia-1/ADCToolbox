"""
Batch runner for all analog output comparison tests.

This module runs all comparison tests for analog output (aout) tools:
- sine_fit
- spec_plot
- spec_plot_phase
- err_hist_sine_code
- err_hist_sine_phase
- basic

Usage:
    pytest run_analog_comparisons.py -v
"""

import pytest
from pathlib import Path

from tests.compare.test_compare_sine_fit import test_compare_sine_fit
from tests.compare.test_compare_spec_plot import test_compare_spec_plot
from tests.compare.test_compare_spec_plot_phase import test_compare_spec_plot_phase
from tests.compare.test_compare_err_hist_sine_code import test_compare_err_hist_sine_code
from tests.compare.test_compare_err_hist_sine_phase import test_compare_err_hist_sine_phase
from tests.compare.test_compare_basic import test_compare_basic


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.parent


def test_analog_basic(project_root):
    """Test basic comparison."""
    test_compare_basic(project_root)


def test_analog_sine_fit(project_root):
    """Test sine_fit comparison."""
    test_compare_sine_fit(project_root)


def test_analog_spec_plot(project_root):
    """Test spec_plot comparison."""
    test_compare_spec_plot(project_root)


def test_analog_spec_plot_phase(project_root):
    """Test spec_plot_phase comparison."""
    test_compare_spec_plot_phase(project_root)


def test_analog_err_hist_sine_code(project_root):
    """Test err_hist_sine_code comparison."""
    test_compare_err_hist_sine_code(project_root)


def test_analog_err_hist_sine_phase(project_root):
    """Test err_hist_sine_phase comparison."""
    test_compare_err_hist_sine_phase(project_root)
