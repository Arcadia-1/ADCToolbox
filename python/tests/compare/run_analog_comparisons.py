"""
Batch runner for all analog output comparison tests.

This module runs all comparison tests for analog output (aout) tools:
- sine_fit
- analyze_spectrum
- analyze_phase_spectrum
- err_hist_sine_code
- err_hist_sine_phase
- err_pdf
- err_spectrum
- err_auto_correlation
- err_envelope_spectrum
- fit_static_nonlin
- inl_sine
- decompose_harmonics
- basic

Usage:
    pytest run_analog_comparisons.py -v
"""

import pytest
from pathlib import Path

from tests.compare.test_compare_sine_fit import test_compare_sine_fit
from tests.compare.test_compare_analyze_spectrum import test_compare_analyze_spectrum
from tests.compare.test_compare_analyze_phase_spectrum import test_compare_analyze_phase_spectrum
from tests.compare.test_compare_err_hist_sine_code import test_compare_err_hist_sine_code
from tests.compare.test_compare_err_hist_sine_phase import test_compare_err_hist_sine_phase
from tests.compare.test_compare_err_pdf import test_compare_err_pdf
from tests.compare.test_compare_err_spectrum import test_compare_err_spectrum
from tests.compare.test_compare_err_auto_correlation import test_compare_err_auto_correlation
from tests.compare.test_compare_err_envelope_spectrum import test_compare_err_envelope_spectrum
from tests.compare.test_compare_fit_static_nonlin import test_compare_fit_static_nonlin
from tests.compare.test_compare_inl_sine import test_compare_inl_sine
from tests.compare.test_compare_decompose_harmonics import test_compare_decompose_harmonics
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


def test_analog_analyze_spectrum(project_root):
    """Test analyze_spectrum comparison."""
    test_compare_analyze_spectrum(project_root)


def test_analog_analyze_phase_spectrum(project_root):
    """Test analyze_phase_spectrum comparison."""
    test_compare_analyze_phase_spectrum(project_root)


def test_analog_err_hist_sine_code(project_root):
    """Test err_hist_sine_code comparison."""
    test_compare_err_hist_sine_code(project_root)


def test_analog_err_hist_sine_phase(project_root):
    """Test err_hist_sine_phase comparison."""
    test_compare_err_hist_sine_phase(project_root)


def test_analog_err_pdf(project_root):
    """Test err_pdf comparison."""
    test_compare_err_pdf(project_root)


def test_analog_err_spectrum(project_root):
    """Test err_spectrum comparison."""
    test_compare_err_spectrum(project_root)


def test_analog_err_auto_correlation(project_root):
    """Test err_auto_correlation comparison."""
    test_compare_err_auto_correlation(project_root)


def test_analog_err_envelope_spectrum(project_root):
    """Test err_envelope_spectrum comparison."""
    test_compare_err_envelope_spectrum(project_root)


def test_analog_fit_static_nonlin(project_root):
    """Test fit_static_nonlin comparison."""
    test_compare_fit_static_nonlin(project_root)


def test_analog_inl_sine(project_root):
    """Test inl_sine comparison."""
    test_compare_inl_sine(project_root)


def test_analog_decompose_harmonics(project_root):
    """Test decompose_harmonics comparison."""
    test_compare_decompose_harmonics(project_root)

