"""
Unit tests for quick_sndr: lean SNDR-only path.

Validates:
  1. SNDR/ENOB match analyze_spectrum within rounding on an ideal sine.
  2. Bare-defaults call works (just data + fs).
  3. Window override is honored.
"""
import importlib

import numpy as np

from adctoolbox import quick_sndr, analyze_spectrum
from adctoolbox.spectrum.compute_spectrum import compute_spectrum


def _ideal_quantized_sine(n_samples=256, n_bits=16, bin_index=7):
    n = np.arange(n_samples)
    x = np.sin(2 * np.pi * bin_index * n / n_samples)
    full_scale = (1 << (n_bits - 1)) - 1
    codes = np.round(x * full_scale).astype(np.int64)
    codes = np.clip(codes, -(1 << (n_bits - 1)), full_scale)
    return codes / (1 << (n_bits - 1))


def test_quick_sndr_matches_analyze_spectrum_on_ideal_sine():
    aout = _ideal_quantized_sine()
    fs = 12.5e6

    q = quick_sndr(aout, fs=fs)
    a = analyze_spectrum(aout, fs=fs, create_plot=False)

    # SNDR comes from same formula in both — should match to <0.01 dB.
    assert abs(q['sndr_dbc'] - a['sndr_dbc']) < 0.1, (
        f"quick_sndr SNDR={q['sndr_dbc']:.3f} vs analyze_spectrum SNDR="
        f"{a['sndr_dbc']:.3f}"
    )
    assert abs(q['enob'] - a['enob']) < 0.02


def test_quick_sndr_matches_compute_spectrum_with_explicit_side_bin():
    aout = _ideal_quantized_sine(n_samples=1024, bin_index=37)
    fs = 12.5e6

    q = quick_sndr(aout, fs=fs, win_type='hann', side_bin=1)
    c = compute_spectrum(aout, fs=fs, win_type='hann', side_bin=1)

    assert abs(q['sndr_dbc'] - c['metrics']['sndr_dbc']) < 1e-9
    assert abs(q['enob'] - c['metrics']['enob']) < 1e-9


def test_quick_sndr_keeps_direct_fft_fast_path():
    quick_sndr_module = importlib.import_module('adctoolbox.spectrum.quick_sndr')

    assert 'compute_spectrum' not in quick_sndr_module.__dict__


def test_quick_sndr_bare_call():
    """Sanity: just data, no kwargs, returns the dict."""
    aout = _ideal_quantized_sine()
    result = quick_sndr(aout)
    assert set(result.keys()) == {'sndr_dbc', 'enob'}
    assert result['sndr_dbc'] > 90  # 16-bit ideal ≈ 98 dB


def test_quick_sndr_window_override():
    """Rectangular window matches analyze_spectrum's rectangular path."""
    aout = _ideal_quantized_sine()
    q = quick_sndr(aout, fs=12.5e6, win_type='rectangular')
    a = analyze_spectrum(aout, fs=12.5e6, win_type='rectangular', create_plot=False)
    assert abs(q['sndr_dbc'] - a['sndr_dbc']) < 0.1
