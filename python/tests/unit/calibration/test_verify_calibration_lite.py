"""Test calibrate_weight_sine_lite function."""

import numpy as np
import time
from adctoolbox.calibration import calibrate_weight_sine_lite
from adctoolbox import analyze_spectrum


def test_calibration_lite():
    """Validate weight recovery from quantized sinewave using lite calibration."""

    # Test configuration
    n_samp = 2**13
    bit_width = 12
    freq_true = 13 / n_samp

    # Generate quantized sinewave
    t = np.arange(n_samp)
    signal = 0.5 * np.sin(2 * np.pi * freq_true * t + np.pi/4) + 0.5
    quantized_signal = np.clip(np.floor(signal * (2**bit_width)), 0, 2**bit_width - 1).astype(int)

    # Extract bits and run calibration
    true_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
    bits = (quantized_signal[:, None] >> np.arange(bit_width - 1, -1, -1)) & 1
    recovered_weights = calibrate_weight_sine_lite(bits, freq=freq_true)

    # Scale recovered weights
    recovered_weights_scaled = recovered_weights * np.max(true_weights)

    # Print weights with equal width formatting
    true_weights_str = ', '.join([f'{w:5.1f}' for w in true_weights])
    recovered_weights_str = ', '.join([f'{w:5.1f}' for w in recovered_weights_scaled])
    print(f"True weights     : [{true_weights_str}]")
    print(f"Recovered weights: [{recovered_weights_str}]")

    # Compute calibrated signal and SNDR
    calibrated_signal = bits @ recovered_weights_scaled
    adc_amplitude = (2**bit_width - 1) / 2.0
    ideal_signal = adc_amplitude * np.sin(2 * np.pi * freq_true * t + np.pi/4) + adc_amplitude
    error_signal = calibrated_signal - ideal_signal

    sndr_before = analyze_spectrum(quantized_signal)['sndr_db']
    sndr_calc = 10 * np.log10(np.mean(ideal_signal**2) / np.mean(error_signal**2))

    np.testing.assert_allclose(sndr_before, sndr_calc, atol=3.0)


def test_calibration_lite_sweep_N():
    """Sweep N from 2^8 to 2^16 and validate calibration."""

    bit_width = 12
    print(f"\n[Batch Test] Sweeping N from 2^8 to 2^16")

    for n_exp in range(3, 17):
        n_samp = 2**n_exp
        freq_true = 3 / n_samp

        # Generate quantized sinewave
        t = np.arange(n_samp)
        signal = 0.5 * np.sin(2 * np.pi * freq_true * t + np.pi/4) + 0.5
        quantized_signal = np.clip(np.floor(signal * (2**bit_width)), 0, 2**bit_width - 1).astype(int)

        # Extract bits and run calibration
        true_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
        bits = (quantized_signal[:, None] >> np.arange(bit_width - 1, -1, -1)) & 1

        start_time = time.time()
        recovered_weights = calibrate_weight_sine_lite(bits, freq=freq_true)
        elapsed_time = time.time() - start_time

        # Scale recovered weights
        recovered_weights_scaled = recovered_weights * np.max(true_weights)

        # Print weights with equal width formatting
        true_weights_str = ', '.join([f'{w:5.1f}' for w in true_weights])
        recovered_weights_str = ', '.join([f'{w:5.1f}' for w in recovered_weights_scaled])
        print(f"True weights     : [{true_weights_str}]")
        print(f"Recovered weights: [{recovered_weights_str}]")

        # Validate weight recovery
        normalized_weights = true_weights / np.max(true_weights)
        max_weight_error = np.max(np.abs(recovered_weights - normalized_weights))
        lsb_threshold = 1 / (2**bit_width)

        # Compute calibrated signal and SNDR
        calibrated_signal = bits @ recovered_weights_scaled
        adc_amplitude = (2**bit_width - 1) / 2.0
        ideal_signal = adc_amplitude * np.sin(2 * np.pi * freq_true * t + np.pi/4) + adc_amplitude
        error_signal = calibrated_signal - ideal_signal

        sndr_before = analyze_spectrum(quantized_signal)['sndr_db']
        sndr_calc = 10 * np.log10(np.mean(ideal_signal**2) / np.mean(error_signal**2))
        sndr_after = analyze_spectrum(calibrated_signal)['sndr_db']

        enob_before = (sndr_before - 1.76) / 6.02
        enob_calc = (sndr_calc - 1.76) / 6.02
        enob_after = (sndr_after - 1.76) / 6.02

        print(f"N=2^{n_exp:2d} ({n_samp:6d}): "
              f"Runtime={elapsed_time*1e3:6.2f}ms, "
              f"Weight_err={max_weight_error:.2e}, "
              f"SNDR: {sndr_before:.1f}/{sndr_calc:.1f}/{sndr_after:.1f} dB, "
              f"ENOB: {enob_before:.2f}/{enob_calc:.2f}/{enob_after:.2f}")

def test_calibration_lite_sweep_fin():
    """Sweep frequency bin from 1 to N/2 - 1."""

    bit_width = 12
    n_samp = 2**13

    print(f"\n[Fin Sweep Test] Sweeping frequency bin from 1 to {n_samp//2 - 1}")
    print(f"N_samples={n_samp}, bit_width={bit_width}")

    for fin_bin in range(1, n_samp // 2, 32):
        freq_true = fin_bin / n_samp

        # Generate quantized sinewave
        t = np.arange(n_samp)
        signal = 0.5 * np.sin(2 * np.pi * freq_true * t + np.pi/4) + 0.5
        quantized_signal = np.clip(np.floor(signal * (2**bit_width)), 0, 2**bit_width - 1).astype(int)

        # Extract bits and run calibration
        true_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
        bits = (quantized_signal[:, None] >> np.arange(bit_width - 1, -1, -1)) & 1

        start_time = time.time()
        recovered_weights = calibrate_weight_sine_lite(bits, freq=freq_true)
        elapsed_time = time.time() - start_time

        # Scale recovered weights
        recovered_weights_scaled = recovered_weights * np.max(true_weights)

        # Print weights with equal width formatting
        true_weights_str = ', '.join([f'{w:5.1f}' for w in true_weights])
        recovered_weights_str = ', '.join([f'{w:5.1f}' for w in recovered_weights_scaled])
        print(f"True weights     : [{true_weights_str}]")
        print(f"Recovered weights: [{recovered_weights_str}]")

        # Validate weight recovery
        normalized_weights = true_weights / np.max(true_weights)
        max_weight_error = np.max(np.abs(recovered_weights - normalized_weights))

        # Compute calibrated signal and SNDR
        calibrated_signal = bits @ recovered_weights_scaled
        adc_amplitude = (2**bit_width - 1) / 2.0
        ideal_signal = adc_amplitude * np.sin(2 * np.pi * freq_true * t + np.pi/4) + adc_amplitude
        error_signal = calibrated_signal - ideal_signal

        sndr_before = analyze_spectrum(quantized_signal)['sndr_db']
        sndr_calc = 10 * np.log10(np.mean(ideal_signal**2) / np.mean(error_signal**2))
        sndr_after = analyze_spectrum(calibrated_signal)['sndr_db']

        enob_before = (sndr_before - 1.76) / 6.02
        enob_calc = (sndr_calc - 1.76) / 6.02
        enob_after = (sndr_after - 1.76) / 6.02

        print(f"Bin={fin_bin:4d}, Freq={freq_true:.6f}: "
              f"Runtime={elapsed_time*1e3:5.2f}ms, "
              f"Weight_err={max_weight_error:.2e}, "
              f"SNDR: {sndr_before:.1f}/{sndr_calc:.1f}/{sndr_after:.1f} dB, "
              f"ENOB: {enob_before:.2f}/{enob_calc:.2f}/{enob_after:.2f}")



def test_calibration_lite_sweep_phase():
    """Sweep phase from 0 to 2π in 16 runs."""

    # Test configuration
    n_samp = 2**13
    bit_width = 12
    freq_true = 13 / n_samp

    print(f"\n[Phase Test] Sweeping phase from 0 to 2π (16 runs)")
    print(f"N_samples={n_samp}, bit_width={bit_width}, freq={freq_true:.6f}")

    n_phases = 36
    for i in range(n_phases + 1):
        phase = i * 2 * np.pi / n_phases

        # Generate quantized sinewave with varying phase
        t = np.arange(n_samp)
        signal = 0.5 * np.sin(2 * np.pi * freq_true * t + phase) + 0.5
        quantized_signal = np.clip(np.floor(signal * (2**bit_width)), 0, 2**bit_width - 1).astype(int)

        # Extract bits and run calibration
        true_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
        bits = (quantized_signal[:, None] >> np.arange(bit_width - 1, -1, -1)) & 1

        start_time = time.time()
        recovered_weights = calibrate_weight_sine_lite(bits, freq=freq_true)
        elapsed_time = time.time() - start_time

        # Scale recovered weights
        recovered_weights_scaled = recovered_weights * np.max(true_weights)

        # Print weights with equal width formatting
        true_weights_str = ', '.join([f'{w:5.1f}' for w in true_weights])
        recovered_weights_str = ', '.join([f'{w:5.1f}' for w in recovered_weights_scaled])
        print(f"True weights     : [{true_weights_str}]")
        print(f"Recovered weights: [{recovered_weights_str}]")

        # Validate weight recovery
        normalized_weights = true_weights / np.max(true_weights)
        max_weight_error = np.max(np.abs(recovered_weights - normalized_weights))

        # Compute calibrated signal and SNDR
        calibrated_signal = bits @ recovered_weights_scaled
        adc_amplitude = (2**bit_width - 1) / 2.0
        ideal_signal = adc_amplitude * np.sin(2 * np.pi * freq_true * t + phase) + adc_amplitude
        error_signal = calibrated_signal - ideal_signal

        sndr_before = analyze_spectrum(quantized_signal)['sndr_db']
        sndr_calc = 10 * np.log10(np.mean(ideal_signal**2) / np.mean(error_signal**2))
        sndr_after = analyze_spectrum(calibrated_signal)['sndr_db']

        enob_before = (sndr_before - 1.76) / 6.02
        enob_calc = (sndr_calc - 1.76) / 6.02
        enob_after = (sndr_after - 1.76) / 6.02

        print(f"Phase={phase:5.3f} rad ({phase*180/np.pi:6.1f}°): "
              f"Runtime={elapsed_time*1e3:5.2f}ms, "
              f"Weight_err={max_weight_error:.2e}, "
              f"SNDR: {sndr_before:.1f}/{sndr_calc:.1f}/{sndr_after:.1f} dB, "
              f"ENOB: {enob_before:.2f}/{enob_calc:.2f}/{enob_after:.2f}")


def test_calibration_lite_sweep_phase_noise():
    """Sweep phase from 0 to 2π in 16 runs."""

    # Test configuration
    n_samp = 2**13
    bit_width = 12
    freq_true = 13 / n_samp

    print(f"\n[Phase Test] Sweeping phase from 0 to 2π (16 runs)")
    print(f"N_samples={n_samp}, bit_width={bit_width}, freq={freq_true:.6f}, noise=0.0002")

    n_phases = 36
    for i in range(n_phases + 1):
        phase = i * 2 * np.pi / n_phases

        # Generate quantized sinewave with varying phase
        t = np.arange(n_samp)
        signal = 0.5 * np.sin(2 * np.pi * freq_true * t + phase) + 0.5 + 0.0002 * np.random.randn(n_samp)
        quantized_signal = np.clip(np.floor(signal * (2**bit_width)), 0, 2**bit_width - 1).astype(int)

        # Extract bits and run calibration
        true_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
        bits = (quantized_signal[:, None] >> np.arange(bit_width - 1, -1, -1)) & 1

        start_time = time.time()
        recovered_weights = calibrate_weight_sine_lite(bits, freq=freq_true)
        elapsed_time = time.time() - start_time

        # Scale recovered weights
        recovered_weights_scaled = recovered_weights * np.max(true_weights)

        # Print weights with equal width formatting
        true_weights_str = ', '.join([f'{w:5.1f}' for w in true_weights])
        recovered_weights_str = ', '.join([f'{w:5.1f}' for w in recovered_weights_scaled])
        print(f"True weights     : [{true_weights_str}]")
        print(f"Recovered weights: [{recovered_weights_str}]")

        # Validate weight recovery
        normalized_weights = true_weights / np.max(true_weights)
        max_weight_error = np.max(np.abs(recovered_weights - normalized_weights))

        # Compute calibrated signal and SNDR
        calibrated_signal = bits @ recovered_weights_scaled
        adc_amplitude = (2**bit_width - 1) / 2.0
        ideal_signal = adc_amplitude * np.sin(2 * np.pi * freq_true * t + phase) + adc_amplitude
        error_signal = calibrated_signal - ideal_signal

        sndr_before = analyze_spectrum(quantized_signal)['sndr_db']
        sndr_calc = 10 * np.log10(np.mean(ideal_signal**2) / np.mean(error_signal**2))
        sndr_after = analyze_spectrum(calibrated_signal)['sndr_db']

        enob_before = (sndr_before - 1.76) / 6.02
        enob_calc = (sndr_calc - 1.76) / 6.02
        enob_after = (sndr_after - 1.76) / 6.02

        print(f"Phase={phase:5.3f} rad ({phase*180/np.pi:6.1f}°): "
              f"Runtime={elapsed_time*1e3:5.2f}ms, "
              f"Weight_err={max_weight_error:.2e}, "
              f"SNDR: {sndr_before:.1f}/{sndr_calc:.1f}/{sndr_after:.1f} dB, "
              f"ENOB: {enob_before:.2f}/{enob_calc:.2f}/{enob_after:.2f}")



def test_calibration_lite_sweep_amplitude():
    """Sweep amplitude from -60 dB to 0 dBFS."""

    # Test configuration
    n_samp = 2**13
    bit_width = 12
    freq_true = 13 / n_samp
    phase = np.pi / 4

    print(f"\n[Amplitude Test] Sweeping amplitude from -60 dB to 0 dBFS")
    print(f"N_samples={n_samp}, bit_width={bit_width}, freq={freq_true:.6f}")

    for amplitude_db in range(-60, 1, 1):
        # Convert dB to linear amplitude (0 dBFS = 0.5 full scale)
        amplitude = 0.5 * 10**(amplitude_db / 20.0)

        # Generate quantized sinewave with varying amplitude
        t = np.arange(n_samp)
        signal = amplitude * np.sin(2 * np.pi * freq_true * t + phase) + 0.5
        quantized_signal = np.clip(np.floor(signal * (2**bit_width)), 0, 2**bit_width - 1).astype(int)

        # Extract bits and run calibration
        true_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
        bits = (quantized_signal[:, None] >> np.arange(bit_width - 1, -1, -1)) & 1
        

        start_time = time.time()
        recovered_weights = calibrate_weight_sine_lite(bits, freq=freq_true)
        elapsed_time = time.time() - start_time

        # Compute calibrated signal directly
        calibrated_signal = bits @ recovered_weights

        # Estimate actual amplitude and offset from calibrated signal
        offset_est = np.mean(calibrated_signal)
        signal_centered = calibrated_signal - offset_est
        amplitude_est = np.std(signal_centered) * np.sqrt(2)

        # Scale recovered weights to match true full-scale weights
        adc_amplitude = 2**bit_width / 2.0
        expected_amplitude = adc_amplitude * amplitude / 0.5
        scale_factor = amplitude_est / expected_amplitude if expected_amplitude > 1e-6 else 1.0
        recovered_weights_scaled = recovered_weights / scale_factor

        # Validate weight recovery
        max_weight_error = np.max(np.abs(recovered_weights_scaled - true_weights))

        # Compute ideal signal at calibrated scale
        ideal_signal = amplitude_est * np.sin(2 * np.pi * freq_true * t + phase) + offset_est
        error_signal = calibrated_signal - ideal_signal

        # Print weights with equal width formatting
        true_weights_str = ', '.join([f'{w:5.1f}' for w in true_weights])
        recovered_weights_str = ', '.join([f'{w:5.1f}' for w in recovered_weights_scaled])
        print(f"True weights     : [{true_weights_str}]")
        print(f"Recovered weights: [{recovered_weights_str}]")

        sndr_before = analyze_spectrum(quantized_signal)['sndr_db']
        sndr_calc = 10 * np.log10(np.mean(ideal_signal**2) / np.mean(error_signal**2))
        sndr_after = analyze_spectrum(calibrated_signal)['sndr_db']

        enob_before = (sndr_before - 1.76) / 6.02
        enob_calc = (sndr_calc - 1.76) / 6.02
        enob_after = (sndr_after - 1.76) / 6.02

        signal_range = quantized_signal.max() - quantized_signal.min()
        print(f"Amplitude={amplitude_db:3d} dBFS (range={signal_range:4d} codes): "
              f"Runtime={elapsed_time*1e3:5.2f}ms, "
              f"Weight_err={max_weight_error:.2e}, "
              f"SNDR: {sndr_before:.1f}/{sndr_calc:.1f}/{sndr_after:.1f} dB, "
              f"ENOB: {enob_before:.2f}/{enob_calc:.2f}/{enob_after:.2f}\n")


def test_calibration_lite_sweep_phase_redundancy():
    """Sweep phase from 0 to 2π with redundant ADC weights."""

    # Test configuration
    n_samp = 2**13
    bit_width = 12
    freq_true = 13 / n_samp

    # Redundant weights: bit 4 and 5 both have weight 128
    true_weights = np.array([2048.0, 1024.0, 512.0, 256.0, 128.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0])

    print(f"\n[Phase Redundancy Test] Sweeping phase from 0 to 2π (36 runs)")
    print(f"N_samples={n_samp}, bit_width={bit_width}, freq={freq_true:.6f}")
    print(f"Redundant weights: {true_weights.tolist()}")

    def decompose_to_redundant_bits(codes, weights):
        """Decompose ADC codes into redundant bit representation using greedy algorithm."""
        n_samples = len(codes)
        bit_width = len(weights)
        bits = np.zeros((n_samples, bit_width), dtype=int)

        for i, code in enumerate(codes):
            remaining = float(code)
            for j, weight in enumerate(weights):
                if remaining >= weight - 0.5:  # tolerance for float comparison
                    bits[i, j] = 1
                    remaining -= weight

        return bits

    n_phases = 36
    for i in range(n_phases + 1):
        phase = i * 2 * np.pi / n_phases

        # Generate quantized sinewave with varying phase
        t = np.arange(n_samp)
        signal = 0.5 * np.sin(2 * np.pi * freq_true * t + phase) + 0.5
        quantized_signal = np.clip(np.floor(signal * (2**bit_width)), 0, 2**bit_width - 1).astype(int)

        # Decompose into redundant bit representation
        bits = decompose_to_redundant_bits(quantized_signal, true_weights)

        start_time = time.time()
        recovered_weights = calibrate_weight_sine_lite(bits, freq=freq_true)
        elapsed_time = time.time() - start_time

        # Scale recovered weights
        recovered_weights_scaled = recovered_weights * np.max(true_weights)

        # Print weights with equal width formatting
        true_weights_str = ', '.join([f'{w:5.1f}' for w in true_weights])
        recovered_weights_str = ', '.join([f'{w:5.1f}' for w in recovered_weights_scaled])
        print(f"True weights     : [{true_weights_str}]")
        print(f"Recovered weights: [{recovered_weights_str}]")

        # Validate weight recovery
        normalized_weights = true_weights / np.max(true_weights)
        max_weight_error = np.max(np.abs(recovered_weights - normalized_weights))

        # Compute calibrated signal and SNDR
        calibrated_signal = bits @ recovered_weights_scaled
        adc_amplitude = (2**bit_width - 1) / 2.0
        ideal_signal = adc_amplitude * np.sin(2 * np.pi * freq_true * t + phase) + adc_amplitude
        error_signal = calibrated_signal - ideal_signal

        sndr_before = analyze_spectrum(quantized_signal)['sndr_db']
        sndr_calc = 10 * np.log10(np.mean(ideal_signal**2) / np.mean(error_signal**2))
        sndr_after = analyze_spectrum(calibrated_signal)['sndr_db']

        enob_before = (sndr_before - 1.76) / 6.02
        enob_calc = (sndr_calc - 1.76) / 6.02
        enob_after = (sndr_after - 1.76) / 6.02

        print(f"Phase={phase:5.3f} rad ({phase*180/np.pi:6.1f}°): "
              f"Runtime={elapsed_time*1e3:5.2f}ms, "
              f"Weight_err={max_weight_error:.2e}, "
              f"SNDR: {sndr_before:.1f}/{sndr_calc:.1f}/{sndr_after:.1f} dB, "
              f"ENOB: {enob_before:.2f}/{enob_calc:.2f}/{enob_after:.2f}")
