import numpy as np
import pytest

from adctoolbox.siggen import ADC_Signal_Generator


def _generator(dc=0.5, n=4096):
    return ADC_Signal_Generator(
        N=n,
        Fs=1000e6,
        Fin=97e6,
        A=0.49,
        DC=dc,
    )


def test_reference_error_zero_droop_is_identity_with_dc_offset():
    gen = _generator(dc=0.5)
    clean = gen.get_clean_signal()

    from_internal = gen.apply_reference_error(
        None,
        settling_tau=2.0,
        droop_strength=0.0,
    )
    from_explicit = gen.apply_reference_error(
        clean,
        settling_tau=2.0,
        droop_strength=0.0,
    )

    np.testing.assert_allclose(from_internal, clean, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(from_explicit, clean, atol=0.0, rtol=0.0)


def test_reference_error_droop_preserves_dc_reference_frame():
    gen = _generator(dc=0.5)
    clean = gen.get_clean_signal()
    out = gen.apply_reference_error(clean, settling_tau=2.0, droop_strength=0.01)

    assert abs(np.mean(out) - gen.DC) < 0.02
    assert np.max(np.abs(out - gen.DC)) < np.max(np.abs(clean - gen.DC))


def test_ra_gain_error_modes_have_explicit_path_semantics():
    gen = _generator(dc=0.5)
    signal = np.linspace(0.01, 0.99, gen.N)
    signal_ac = signal - gen.DC
    msb_bits = 4
    lsb_bits = 12
    gain = 0.99

    msb = np.floor(signal_ac * 2**msb_bits) / 2**msb_bits
    lsb = np.floor((signal_ac - msb) * 2**lsb_bits) / 2**lsb_bits

    coarse_gain = gen.apply_ra_gain_error(
        signal,
        relative_gain=gain,
        msb_bits=msb_bits,
        lsb_bits=lsb_bits,
        mode="coarse_path",
    )
    residue_gain = gen.apply_ra_gain_error(
        signal,
        relative_gain=gain,
        msb_bits=msb_bits,
        lsb_bits=lsb_bits,
        mode="residue_path",
    )
    nominal = gen.apply_ra_gain_error(
        signal,
        relative_gain=1.0,
        msb_bits=msb_bits,
        lsb_bits=lsb_bits,
    )

    np.testing.assert_allclose(coarse_gain - nominal, (gain - 1.0) * msb)
    np.testing.assert_allclose(residue_gain - nominal, (gain - 1.0) * lsb)


def test_ra_gain_error_rejects_unknown_mode():
    gen = _generator()
    with pytest.raises(ValueError, match="mode"):
        gen.apply_ra_gain_error(mode="residue_amp")
    with pytest.raises(ValueError, match="mode"):
        gen.apply_ra_gain_error_dynamic(mode="residue_amp")


def test_dynamic_ra_gain_error_modes_are_distinct_when_gain_changes():
    gen = _generator(n=128)
    signal = np.linspace(0.01, 0.99, gen.N)

    coarse = gen.apply_ra_gain_error_dynamic(
        signal,
        relative_gain=0.99,
        coeff_3=0.0,
        mode="coarse_path",
    )
    residue = gen.apply_ra_gain_error_dynamic(
        signal,
        relative_gain=0.99,
        coeff_3=0.0,
        mode="residue_path",
    )

    assert not np.allclose(coarse, residue)


def test_dynamic_ra_gain_error_coarse_path_preserves_legacy_formula():
    gen = _generator(n=128)
    signal = np.linspace(0.01, 0.99, gen.N)
    signal_ac = signal - gen.DC
    relative_gain = 1.0
    coeff_3 = 0.15
    msb_bits = 4
    lsb_bits = 12
    expected_ac = np.zeros(gen.N)
    prev_output_ac = 0.0

    for n, v_in_ac in enumerate(signal_ac):
        msb = np.floor(v_in_ac * 2**msb_bits) / 2**msb_bits
        lsb = np.floor((v_in_ac - msb) * 2**lsb_bits) / 2**lsb_bits
        dynamic_gain = relative_gain + coeff_3 * (prev_output_ac**2)
        expected_ac[n] = msb * dynamic_gain + lsb
        prev_output_ac = expected_ac[n]

    actual = gen.apply_ra_gain_error_dynamic(
        signal,
        relative_gain=relative_gain,
        coeff_3=coeff_3,
        msb_bits=msb_bits,
        lsb_bits=lsb_bits,
    )

    np.testing.assert_allclose(actual, expected_ac + gen.DC)
