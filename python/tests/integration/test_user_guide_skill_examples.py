"""Smoke tests that actually run every code example shown in the bundled
`adctoolbox-user-guide` skill.

Background — Issue #11 + the user-guide-skill-revamp PR (#13) shipped doc
examples that didn't match the real API (made-up params like `Fin=`,
`n_bits=`; wrong return keys like `metrics["SNDR"]` instead of
`metrics["sndr_dbc"]`). The eval used during the revamp only scored text
similarity from subagents — nothing actually imported and called the
code.

These tests are the real fence: if a future SKILL.md / advanced-debug.md
example diverges from the API again, this file fails.

Each test mirrors a code block from one of:
- `python/src/adctoolbox/_bundled_skills/skills/adctoolbox-user-guide/SKILL.md`
- `.../adctoolbox-user-guide/references/advanced-debug.md`
- `.../adctoolbox-user-guide/references/api-quickref.md`
"""

import numpy as np
import pytest


# ----------------------------------------------------------------------
# Shared fixture — a small synthetic dout/aout pair used by most tests
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_capture():
    """8-bit, 4096-sample coherent sine — small enough to be fast,
    large enough that all analysis APIs accept it."""
    from adctoolbox.siggen import ADC_Signal_Generator
    from adctoolbox import find_coherent_frequency

    n_samples = 4096
    fs = 100e6
    n_bits = 8
    fin_target = 5e6
    fin_actual, _ = find_coherent_frequency(fs, fin_target, n_samples)

    gen = ADC_Signal_Generator(N=n_samples, Fs=fs, Fin=fin_actual, A=0.45, DC=0.5)
    aout = gen.get_clean_signal()
    aout = gen.apply_thermal_noise(noise_rms=1e-4, input_signal=aout)
    dout_codes_signal = gen.apply_quantization_noise(
        n_bits=n_bits, quant_range=(0.0, 1.0), input_signal=aout
    )

    code_levels = np.round((dout_codes_signal - 0.0) / (1.0 / (2 ** n_bits))).astype(int)
    code_levels = np.clip(code_levels, 0, 2 ** n_bits - 1)
    bits = np.array([
        [(c >> b) & 1 for b in range(n_bits - 1, -1, -1)]
        for c in code_levels
    ])

    return {
        "aout": aout.astype(float),
        "bits": bits,
        "fs": fs,
        "fin_actual": fin_actual,
        "n_bits": n_bits,
        "n_samples": n_samples,
        "freq_norm": fin_actual / fs,
    }


# ----------------------------------------------------------------------
# SKILL.md §3 — basic spectrum workflow
# ----------------------------------------------------------------------

def test_skill_md_section_3_spectrum_workflow(synth_capture):
    from adctoolbox import (
        analyze_spectrum,
        analyze_spectrum_polar,
        find_coherent_frequency,
        fit_sine_4param,
    )
    from adctoolbox.fundamentals import validate_aout_data, validate_dout_data
    from adctoolbox.spectrum import compute_spectrum

    aout = synth_capture["aout"]
    bits = synth_capture["bits"]
    fs = synth_capture["fs"]
    n_samples = synth_capture["n_samples"]

    validate_aout_data(aout)
    validate_dout_data(bits)

    fin_hz, k = find_coherent_frequency(fs, 5e6, n_samples)
    assert isinstance(k, (int, np.integer))
    assert fin_hz > 0

    metrics = analyze_spectrum(aout, fs=fs, create_plot=False)
    assert {"sndr_dbc", "sfdr_dbc", "enob"} <= set(metrics.keys())
    assert metrics["enob"] > 0

    polar = analyze_spectrum_polar(aout, fs=fs, create_plot=False)
    assert "sndr_dbc" in polar

    full = compute_spectrum(aout, fs=fs)
    assert {"metrics", "plot_data"} <= set(full.keys())
    assert "freq" in full["plot_data"]

    fit = fit_sine_4param(aout)
    assert 0.0 <= fit["frequency"] <= 0.5
    assert {"amplitude", "phase", "dc_offset", "rmse"} <= set(fit.keys())


# ----------------------------------------------------------------------
# SKILL.md §4 — basic digital calibration
# ----------------------------------------------------------------------

def test_skill_md_section_4_calibration(synth_capture):
    from adctoolbox import calibrate_weight_sine
    from adctoolbox.calibration import calibrate_weight_sine_lite

    bits = synth_capture["bits"]
    freq_norm = synth_capture["freq_norm"]

    full = calibrate_weight_sine(bits, freq=freq_norm)
    assert "weight" in full
    assert isinstance(full["weight"], np.ndarray)

    fast = calibrate_weight_sine_lite(bits, freq_norm)
    assert isinstance(fast, np.ndarray)
    assert fast.ndim == 1
    assert len(fast) == bits.shape[1]


# ----------------------------------------------------------------------
# advanced-debug.md — dashboards
# ----------------------------------------------------------------------

def test_advanced_dashboards(synth_capture, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from adctoolbox.toolset import generate_aout_dashboard, generate_dout_dashboard

    aout = synth_capture["aout"]
    bits = synth_capture["bits"]
    fs = synth_capture["fs"]
    fin_actual = synth_capture["fin_actual"]
    freq_norm = synth_capture["freq_norm"]

    fig_a, axes_a = generate_aout_dashboard(
        aout, fs=fs, freq=fin_actual, output_path=str(tmp_path / "aout.png")
    )
    fig_d, axes_d = generate_dout_dashboard(
        bits, freq=freq_norm, output_path=str(tmp_path / "dout.png")
    )
    assert (tmp_path / "aout.png").exists()
    assert (tmp_path / "dout.png").exists()


# ----------------------------------------------------------------------
# advanced-debug.md — phase-plane
# ----------------------------------------------------------------------

def test_advanced_phase_plane(synth_capture):
    import matplotlib
    matplotlib.use("Agg")
    from adctoolbox.aout import analyze_phase_plane, analyze_error_phase_plane

    aout = synth_capture["aout"]
    fs = synth_capture["fs"]

    pp = analyze_phase_plane(aout, fs=fs, create_plot=False)
    assert "lag" in pp and "outliers" in pp

    epp = analyze_error_phase_plane(aout, fs=fs, create_plot=False)
    assert {"residual", "fitted_params"} <= set(epp.keys())


# ----------------------------------------------------------------------
# advanced-debug.md — bit-level / overflow / enob sweep / radix
# ----------------------------------------------------------------------

def test_advanced_bit_level(synth_capture):
    import matplotlib
    matplotlib.use("Agg")
    from adctoolbox import (
        analyze_bit_activity,
        analyze_overflow,
        analyze_enob_sweep,
        analyze_weight_radix,
        calibrate_weight_sine,
    )

    bits = synth_capture["bits"]
    freq_norm = synth_capture["freq_norm"]

    activity = analyze_bit_activity(bits, create_plot=False)
    assert isinstance(activity, np.ndarray)
    assert activity.shape == (bits.shape[1],)

    weight_dict = calibrate_weight_sine(bits, freq=freq_norm)
    weights = weight_dict["weight"]

    rmin, rmax, ovf0, ovf1 = analyze_overflow(bits, weights, create_plot=False)
    assert all(isinstance(arr, np.ndarray) for arr in (rmin, rmax, ovf0, ovf1))

    enob_sweep, n_bits_vec = analyze_enob_sweep(
        bits, freq=freq_norm, create_plot=False
    )
    assert enob_sweep.shape == n_bits_vec.shape

    radix_info = analyze_weight_radix(weights, create_plot=False)
    assert {"radix", "wgtsca", "effres"} <= set(radix_info.keys())


# ----------------------------------------------------------------------
# advanced-debug.md — static nonlinearity
# ----------------------------------------------------------------------

def test_advanced_static_nonlinearity(synth_capture):
    from adctoolbox import fit_static_nonlin

    aout = synth_capture["aout"]

    k2, k3, fitted_sine, fitted_transfer = fit_static_nonlin(aout, order=3)
    assert isinstance(fitted_sine, np.ndarray)
    assert isinstance(fitted_transfer, tuple)
    assert len(fitted_transfer) == 2


# ----------------------------------------------------------------------
# advanced-debug.md — cap-to-weight
# ----------------------------------------------------------------------

def test_advanced_cap_to_weight():
    from adctoolbox.fundamentals import convert_cap_to_weight

    n_bits = 6
    caps_bit = np.array([2 ** i for i in range(n_bits)], dtype=float)
    caps_bridge = np.zeros(n_bits, dtype=float)
    caps_parasitic = np.zeros(n_bits, dtype=float)

    weights, c_total = convert_cap_to_weight(caps_bit, caps_bridge, caps_parasitic)
    assert isinstance(weights, np.ndarray)
    assert c_total > 0


# ----------------------------------------------------------------------
# api-quickref.md — flat-export presence (one big import sanity check)
# ----------------------------------------------------------------------

def test_api_quickref_flat_imports_resolve():
    from adctoolbox import (
        analyze_spectrum, analyze_spectrum_polar,
        find_coherent_frequency, fit_sine_4param,
        calibrate_weight_sine,
        analyze_error_by_value, analyze_error_by_phase,
        analyze_error_pdf, analyze_error_spectrum,
        analyze_error_autocorr, analyze_error_envelope_spectrum,
        analyze_inl_from_sine,
        analyze_decomposition_time, analyze_decomposition_polar,
        fit_static_nonlin,
        analyze_bit_activity, analyze_overflow,
        analyze_weight_radix, analyze_enob_sweep,
        plot_residual_scatter,
        calculate_walden_fom, calculate_schreier_fom,
        calculate_thermal_noise_limit, calculate_jitter_limit,
        db_to_mag, mag_to_db, db_to_power, power_to_db,
        snr_to_enob, enob_to_snr, snr_to_nsd, nsd_to_snr,
        bin_to_freq, freq_to_bin, fold_frequency_to_nyquist,
        ntf_analyzer,
    )
    # Each should be a callable / class
    locals_dict = locals()
    for name, obj in locals_dict.items():
        assert callable(obj), f"{name} is not callable"


def test_api_quickref_submodule_imports_resolve():
    from adctoolbox.siggen import ADC_Signal_Generator
    from adctoolbox.calibration import calibrate_weight_sine_lite
    from adctoolbox.fundamentals import (
        validate_aout_data, validate_dout_data, convert_cap_to_weight
    )
    from adctoolbox.aout import analyze_phase_plane, analyze_error_phase_plane
    from adctoolbox.spectrum import compute_spectrum
    from adctoolbox.toolset import (
        generate_aout_dashboard, generate_dout_dashboard
    )
    for obj in (
        ADC_Signal_Generator, calibrate_weight_sine_lite,
        validate_aout_data, validate_dout_data, convert_cap_to_weight,
        analyze_phase_plane, analyze_error_phase_plane,
        compute_spectrum, generate_aout_dashboard, generate_dout_dashboard,
    ):
        assert callable(obj)
