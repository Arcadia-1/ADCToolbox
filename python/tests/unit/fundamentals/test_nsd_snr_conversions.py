"""Unit tests for NSD/SNR conversion functions."""

import numpy as np
import pytest
from adctoolbox.common import snr_to_nsd, nsd_to_snr


class TestNsdSnrConversions:
    """Test suite for NSD and SNR conversion functions."""

    def test_snr_to_nsd_basic(self):
        """Test basic SNR to NSD conversion."""
        snr_db = 80
        fs = 1e6
        osr = 1
        nsd = snr_to_nsd(snr_db, fs, signal_pwr_dbfs=0, osr=osr)

        # For OSR=1, BW = fs/2 = 500 kHz
        # NSD = 0 - 80 - 10*log10(500000) = -136.99 dBFS/Hz
        expected = 0 - 80 - 10 * np.log10(fs / 2)
        assert np.isclose(nsd, expected, rtol=1e-10)

    def test_nsd_to_snr_basic(self):
        """Test basic NSD to SNR conversion."""
        nsd_dbfs_hz = -140
        fs = 10e6
        osr = 1
        snr = nsd_to_snr(nsd_dbfs_hz, fs, signal_pwr_dbfs=0, osr=osr)

        # For OSR=1, BW = fs/2 = 5 MHz
        # SNR = 0 - (-140 + 10*log10(5e6)) = 73.01 dB
        expected = 0 - (nsd_dbfs_hz + 10 * np.log10(fs / 2))
        assert np.isclose(snr, expected, rtol=1e-10)

    def test_snr_to_nsd_with_osr(self):
        """Test SNR to NSD conversion with oversampling."""
        snr_db = 80
        fs = 1e6
        osr = 256
        nsd = snr_to_nsd(snr_db, fs, signal_pwr_dbfs=0, osr=osr)

        # For OSR=256, BW = fs/(2*256) = 1953.125 Hz
        # NSD = 0 - 80 - 10*log10(1953.125) = -112.91 dBFS/Hz
        bw = fs / (2 * osr)
        expected = 0 - 80 - 10 * np.log10(bw)
        assert np.isclose(nsd, expected, rtol=1e-10)

    def test_nsd_to_snr_with_osr(self):
        """Test NSD to SNR conversion with oversampling."""
        nsd_dbfs_hz = -140
        fs = 10e6
        osr = 256
        snr = nsd_to_snr(nsd_dbfs_hz, fs, signal_pwr_dbfs=0, osr=osr)

        # For OSR=256, BW = fs/(2*256) = 19531.25 Hz
        # SNR = 0 - (-140 + 10*log10(19531.25)) = 97.09 dB
        bw = fs / (2 * osr)
        expected = 0 - (nsd_dbfs_hz + 10 * np.log10(bw))
        assert np.isclose(snr, expected, rtol=1e-10)

    def test_round_trip_conversion(self):
        """Test that converting SNR->NSD->SNR returns original value."""
        snr_original = 85.3
        fs = 1e6
        osr = 128

        nsd = snr_to_nsd(snr_original, fs, osr=osr)
        snr_recovered = nsd_to_snr(nsd, fs, osr=osr)

        assert np.isclose(snr_original, snr_recovered, rtol=1e-10)

    def test_round_trip_conversion_reverse(self):
        """Test that converting NSD->SNR->NSD returns original value."""
        nsd_original = -135.5
        fs = 5e6
        osr = 64

        snr = nsd_to_snr(nsd_original, fs, osr=osr)
        nsd_recovered = snr_to_nsd(snr, fs, osr=osr)

        assert np.isclose(nsd_original, nsd_recovered, rtol=1e-10)

    def test_with_non_zero_signal_power(self):
        """Test conversions with non-zero signal power."""
        snr_db = 70
        fs = 1e6
        osr = 1
        signal_pwr_dbfs = -6  # -6 dBFS signal

        nsd = snr_to_nsd(snr_db, fs, signal_pwr_dbfs=signal_pwr_dbfs, osr=osr)
        snr_recovered = nsd_to_snr(nsd, fs, signal_pwr_dbfs=signal_pwr_dbfs, osr=osr)

        assert np.isclose(snr_db, snr_recovered, rtol=1e-10)

    def test_array_input(self):
        """Test that functions work with array inputs."""
        snr_array = np.array([70, 80, 90])
        fs = 1e6
        osr = 16

        nsd_array = snr_to_nsd(snr_array, fs, osr=osr)
        assert nsd_array.shape == snr_array.shape

        snr_recovered = nsd_to_snr(nsd_array, fs, osr=osr)
        assert np.allclose(snr_array, snr_recovered, rtol=1e-10)

    def test_osr_effect_on_snr(self):
        """Test that doubling OSR increases SNR by ~3 dB (10*log10(2))."""
        nsd = -140
        fs = 10e6

        snr_osr1 = nsd_to_snr(nsd, fs, osr=1)
        snr_osr2 = nsd_to_snr(nsd, fs, osr=2)

        # Doubling OSR should increase SNR by 10*log10(2) ≈ 3.01 dB
        expected_difference = 10 * np.log10(2)
        assert np.isclose(snr_osr2 - snr_osr1, expected_difference, rtol=1e-10)

    def test_osr_effect_on_nsd(self):
        """Test that changing OSR doesn't affect NSD calculation consistency."""
        snr = 80
        fs = 1e6

        nsd_osr1 = snr_to_nsd(snr, fs, osr=1)
        nsd_osr256 = snr_to_nsd(snr, fs, osr=256)

        # NSD should change by 10*log10(256) = 24.08 dB
        expected_difference = 10 * np.log10(256)
        assert np.isclose(nsd_osr256 - nsd_osr1, expected_difference, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
