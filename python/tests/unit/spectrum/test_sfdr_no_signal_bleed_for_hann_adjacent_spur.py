"""
Regression test for SFDR-vs-side-bin signal-lobe bleed bug.

Bug (fixed 2026-05-13): when the highest spur lands at a bin adjacent to
the fundamental's main lobe (e.g. spur at bin 9 when Hann fundamental
covers bins [6..8]), the spur-power summation in `_extract_highest_spur`
used the unmasked `spectrum_power[spur_start:spur_end]` instead of the
fundamental-zeroed `spectrum_copy[spur_start:spur_end]`.  With Hann +
side_bin=1, the spur summation window [8, 11) included bin 8 (-6 dBc side
of fundamental), polluting the spur power and crashing the reported SFDR
to ~7.78 dB for a signal that's actually 16-bit ideal (~108 dB SFDR).

This test reproduces the failure mode and asserts the corrected behavior.
"""
import numpy as np
from scipy.signal import windows

from adctoolbox import analyze_spectrum


def _ideal_quantized_sine(n_samples: int, n_bits: int, bin_index: int) -> np.ndarray:
    """Coherent sine at `bin_index`, full-scale, rounded to n_bits codes."""
    n = np.arange(n_samples)
    x = np.sin(2 * np.pi * bin_index * n / n_samples)
    full_scale = (1 << (n_bits - 1)) - 1
    codes = np.round(x * full_scale).astype(np.int64)
    codes = np.clip(codes, -(1 << (n_bits - 1)), full_scale)
    return codes / (1 << (n_bits - 1))


def test_sfdr_no_signal_bleed_for_hann_adjacent_spur():
    """
    Coherent 16-bit sine, Hann window (default).  Pre-fix this returned
    SFDR ≈ 7.78 dBc because spur sum at bin 9 reached back to bin 8.  Post-fix
    should be > 100 dBc, on par with rectangular-window analysis of the same
    signal.
    """
    fs = 12.5e6
    aout = _ideal_quantized_sine(n_samples=256, n_bits=16, bin_index=7)

    m_hann = analyze_spectrum(aout, fs=fs, create_plot=False)              # win_type='hann' default
    m_rect = analyze_spectrum(aout, fs=fs, win_type='rectangular', create_plot=False)

    print(f"  Hann SFDR = {m_hann['sfdr_dbc']:.2f} dBc")
    print(f"  Rect SFDR = {m_rect['sfdr_dbc']:.2f} dBc")

    # Pre-fix bug produced SFDR ≈ 7.78 dB.  Post-fix should be > 100 dB.
    assert m_hann['sfdr_dbc'] > 100.0, (
        f"Hann SFDR={m_hann['sfdr_dbc']:.2f} dBc indicates spur-sum window "
        f"is bleeding into fundamental main lobe — _harmonics.py:196 should "
        f"use spectrum_copy (signal-zeroed), not spectrum_power."
    )
    # Both windows should agree within ~10 dB on an ideal sine
    assert abs(m_hann['sfdr_dbc'] - m_rect['sfdr_dbc']) < 10.0, (
        f"Hann/Rect SFDR disagree: hann={m_hann['sfdr_dbc']:.2f}, "
        f"rect={m_rect['sfdr_dbc']:.2f}"
    )
