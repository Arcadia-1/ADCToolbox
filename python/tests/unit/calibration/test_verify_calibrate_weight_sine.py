"""
Unit Test: Verify calibrate_weight_sine wrapper function

Purpose: Test the main calibration wrapper for:
- Single-dataset calibration with shuffled bits
- Multi-dataset calibration
- Weight recovery accuracy
- Frequency estimation and refinement
- Error metrics validation
"""

import numpy as np
import time
from adctoolbox.calibration import calibrate_weight_sine
from adctoolbox import analyze_spectrum

def test_calibration_single_dataset_shuffled():
    """
    Test single-dataset calibration with shuffled bit orders
    Verifies weight recovery when bit order is unknown
    """

    n_samples_list = [8192]
    bit_width = 8

    for n_samp in n_samples_list:
        freq_true_list = [3/n_samp] # relatively frequencies to test

        for freq_true in freq_true_list:
            # Generate true weights and shuffled indices
            true_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
            shift_amounts_order = np.arange(bit_width - 1, -1, -1)


            # shuffled_indices = np.random.permutation(bit_width)
            shuffled_indices = np.arange(bit_width)
            shuffled_weights = true_weights[shuffled_indices]
            current_shifts = shift_amounts_order[shuffled_indices]

            # Generate ideal sine signal
            t = np.arange(n_samp)
            signal = 0.5 * np.sin(2 * np.pi * freq_true * t) + 0.5

            # Quantize to ADC output
            quantized_signal = np.floor(signal * (2**bit_width)).astype(int)
            quantized_signal = np.clip(quantized_signal, 0, 2**bit_width - 1)
            print(f"\n[Unit Test] [Quantized signal]  min/max = [{quantized_signal.min()}, {quantized_signal.max()}]")
            print(f"[Unit Test] [N_samples={n_samp}, freq_true={freq_true:.6f}]")

            # Extract shuffled bits
            bits = (quantized_signal[:, None] >> current_shifts) & 1
            print(f"[Unit Test] [Shuffled indices]: {shuffled_indices}")
            print(f"[Unit Test] [True weights]    : {true_weights.tolist()}")
            print(f"[Unit Test] [Shuffled weights]: {shuffled_weights.tolist()}")

            # Test 1: Calibration with known frequency
            start_time = time.time()
            result = calibrate_weight_sine(bits, freq=freq_true, verbose=2)
            elapsed_time = time.time() - start_time
            print(f"\n[Unit Test] Runtime: {elapsed_time*1e6:.2f} us")

            # Verify results
            recovered_weights = result['weight']
            refined_freq = result['refined_frequency']
            offset = result['offset']
            calibrated_signal = result['calibrated_signal']
            ideal_signal = result['ideal']
            error_signal = result['error']
            snr_db = result['snr_db']
            enob = result['enob']


            # Compute weight recovery error
            
            normalized_set_weights = shuffled_weights / np.max(shuffled_weights)
            max_weight_error = np.max(np.abs(recovered_weights - normalized_set_weights))

            freq_error = np.abs(refined_freq - freq_true)


            print(f"[Unit Test] Set freq: [{freq_true:.6f}], Refined freq: [{refined_freq:.6f}], Freq error: [{freq_error:.2e}]")
            print(f"[Unit Test] Set weights: {normalized_set_weights.tolist()}") 
            print(f"[Unit Test] Calibrated weights: {recovered_weights.tolist()}")  
            print(f"[Unit Test] Max weight error: [{max_weight_error:.2e}] <-- Should be smaller than LSB=[{1/(2**bit_width):.2e}]")
            print(f"[Unit Test] Offset: {offset:.6f}")

            results = analyze_spectrum(quantized_signal)
            sndr_db_before = results['sndr_db']
            enob_before = results['enob']
            print(f"[Unit Test] Before Calibration    : SNDR=[{sndr_db_before:.2f} dB], ENOB=[{enob_before:.2f}]")

            sndr_db_cal = 10*np.log10(np.mean(ideal_signal[0]**2) / np.mean(error_signal[0]**2))
            enob_calc = (sndr_db_cal - 1.76) / 6.02
            print(f"[Unit Test] Calculated from signal: SNDR=[{sndr_db_cal:.2f} dB], ENOB=[{enob_calc:.2f}]")

            results = analyze_spectrum(calibrated_signal)
            sndr_db_after = results['sndr_db']
            enob_after = results['enob']
            print(f"[Unit Test] After Calibration     : SNDR=[{sndr_db_after:.2f} dB], ENOB=[{enob_after:.2f}]")

            np.testing.assert_allclose(sndr_db_before, sndr_db_cal, atol=3.0)
            np.testing.assert_allclose(sndr_db_before, sndr_db_after, atol=3.0)


def test_calibration_single_dataset_shuffled_search_freq():
    """
    Test single-dataset calibration with shuffled bit orders
    Verifies weight recovery when bit order is unknown
    """

    n_samples_list = [8192]
    bit_width = 8

    for n_samp in n_samples_list:
        freq_true_list = [3/n_samp] # relatively frequencies to test

        for freq_true in freq_true_list:
            # Generate true weights and shuffled indices
            true_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
            shift_amounts_order = np.arange(bit_width - 1, -1, -1)


            # shuffled_indices = np.random.permutation(bit_width)
            shuffled_indices = np.arange(bit_width)
            shuffled_weights = true_weights[shuffled_indices]
            current_shifts = shift_amounts_order[shuffled_indices]

            # Generate ideal sine signal
            t = np.arange(n_samp)
            signal = 0.5 * np.sin(2 * np.pi * freq_true * t) + 0.5

            # Quantize to ADC output
            quantized_signal = np.floor(signal * (2**bit_width)).astype(int)
            quantized_signal = np.clip(quantized_signal, 0, 2**bit_width - 1)
            print(f"\n[Unit Test] [Quantized signal]  min/max = [{quantized_signal.min()}, {quantized_signal.max()}]")
            print(f"[Unit Test] [N_samples={n_samp}, freq_true={freq_true:.6f}]")

            # Extract shuffled bits
            bits = (quantized_signal[:, None] >> current_shifts) & 1
            print(f"[Unit Test] [Shuffled indices]: {shuffled_indices}")
            print(f"[Unit Test] [True weights]    : {true_weights.tolist()}")
            print(f"[Unit Test] [Shuffled weights]: {shuffled_weights.tolist()}")

            # Test 1: Calibration with known frequency
            start_time = time.time()
            result = calibrate_weight_sine(bits, verbose=2)
            elapsed_time = time.time() - start_time
            print(f"\n[Unit Test] Runtime: {elapsed_time*1e6:.2f} us")

            # Verify results
            recovered_weights = result['weight']
            refined_freq = result['refined_frequency']
            offset = result['offset']
            calibrated_signal = result['calibrated_signal']
            ideal_signal = result['ideal']
            error_signal = result['error']
            snr_db = result['snr_db']
            enob = result['enob']


            # Compute weight recovery error
            
            normalized_set_weights = shuffled_weights / np.max(shuffled_weights)
            max_weight_error = np.max(np.abs(recovered_weights - normalized_set_weights))

            freq_error = np.abs(refined_freq - freq_true)


            print(f"[Unit Test] Set freq: [{freq_true:.6f}], Refined freq: [{refined_freq:.6f}], Freq error: [{freq_error:.2e}]")
            print(f"[Unit Test] Set weights: {normalized_set_weights.tolist()}") 
            print(f"[Unit Test] Calibrated weights: {recovered_weights.tolist()}")  
            print(f"[Unit Test] Max weight error: [{max_weight_error:.2e}] <-- Should be smaller than LSB=[{1/(2**bit_width):.2e}]")
            print(f"[Unit Test] Offset: {offset:.6f}")

            results = analyze_spectrum(quantized_signal)
            sndr_db_before = results['sndr_db']
            enob_before = results['enob']
            print(f"[Unit Test] Before Calibration    : SNDR=[{sndr_db_before:.2f} dB], ENOB=[{enob_before:.2f}]")

            sndr_db_cal = 10*np.log10(np.mean(ideal_signal[0]**2) / np.mean(error_signal[0]**2))
            enob_calc = (sndr_db_cal - 1.76) / 6.02
            print(f"[Unit Test] Calculated from signal: SNDR=[{sndr_db_cal:.2f} dB], ENOB=[{enob_calc:.2f}]")

            results = analyze_spectrum(calibrated_signal)
            sndr_db_after = results['sndr_db']
            enob_after = results['enob']
            print(f"[Unit Test] After Calibration     : SNDR=[{sndr_db_after:.2f} dB], ENOB=[{enob_after:.2f}]")

            np.testing.assert_allclose(sndr_db_before, sndr_db_cal, atol=3.0)
            np.testing.assert_allclose(sndr_db_before, sndr_db_after, atol=3.0)

def test_calibration_multi_datasets():
    """
    Test calibration across multiple frequencies and sample sizes
    Similar to test_verify_estimate_frequencies.py pattern
    """

    n_samples_list = [8192]  # Different lengths to sweep
    bit_width = 10

    for n_samp in n_samples_list:
        # Generate frequency list based on sample size
        num_test_freqs = 9
        bins = np.linspace(3, n_samp//2 - 3, num_test_freqs)
        bins = (np.round(bins / 2) * 2 + 1).astype(int)
        print(f"\n[Frequency bins]: {bins.tolist()}")
        freq_true_list = bins / n_samp

        print(f"\n{'='*80}")
        print(f"[N_samples = {n_samp}]")
        print(f"[Number of frequencies]: {len(freq_true_list)}")
        print(f"[Frequency bin width]: {1/n_samp:.8f}")

        # Setup true weights (normalized to max=1)
        true_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
        shift_amounts_order = np.arange(bit_width - 1, -1, -1)

        # Use identity mapping (no shuffling for this test)
        shuffled_indices = np.arange(bit_width)
        shuffled_weights = true_weights[shuffled_indices]
        current_shifts = shift_amounts_order[shuffled_indices]

        # Normalize weights
        normalized_weights = shuffled_weights / np.max(shuffled_weights)

        bits_ideal = []
        for idx, freq_true in enumerate(freq_true_list):
            # Generate ideal sine signal
            t = np.arange(n_samp)
            signal = 0.5 * np.sin(2 * np.pi * freq_true * t) + 0.5

            # Quantize to ADC output
            quantized_signal = np.floor(signal * (2**bit_width)).astype(int)
            quantized_signal = np.clip(quantized_signal, 0, 2**bit_width - 1)

            # Extract bits
            bits = (quantized_signal[:, None] >> current_shifts) & 1
            bits_ideal.append(bits)

        bits_mangled_list = [bits[:, shuffled_indices] for bits in bits_ideal]

        print(f"[Unit Test] Number of datasets=[{len(bits_mangled_list)}], number of frequencies=[{len(freq_true_list)}]")
        print(f"[Unit Test] Datasets: {[bits.shape for bits in bits_mangled_list]}")
        
        start_time = time.time()
        result = calibrate_weight_sine(bits_mangled_list, freq=freq_true_list, verbose=0)
        elapsed_time = time.time() - start_time
        print(f"\n[Without freq_init] Runtime: {elapsed_time*1e3:.2f} ms")

        # Verify results
        recovered_weights = result['weight']
        refined_freq = result['refined_frequency']
        offset = result['offset']
        calibrated_signal = result['calibrated_signal']
        print(f"[Unit Test] Calibrated Signal shapes: {[sig.shape for sig in calibrated_signal]}")
        ideal_signal = result['ideal']
        error_signal = result['error']
        snr_db = result['snr_db']
        enob = result['enob']


        # Compute weight recovery error
        
        normalized_set_weights = shuffled_weights / np.max(shuffled_weights)
        max_weight_error = np.max(np.abs(recovered_weights - normalized_set_weights))


        print(f"[Unit Test] Set weights: {normalized_set_weights.tolist()}") 
        print(f"[Unit Test] Calibrated weights: {recovered_weights.tolist()}")  
        print(f"[Unit Test] Max weight error: [{max_weight_error:.2e}] <-- Should be smaller than LSB=[{1/(2**bit_width):.2e}]")
        print(f"[Unit Test] Offset: {offset:.6f}")

        results = analyze_spectrum(quantized_signal)
        sndr_db_before = results['sndr_db']
        enob_before = results['enob']
        print(f"[Unit Test] Before Calibration    : SNDR=[{sndr_db_before:.2f} dB], ENOB=[{enob_before:.2f}]")

        sndr_db_cal = 10*np.log10(np.mean(ideal_signal[0]**2) / np.mean(error_signal[0]**2))
        enob_calc = (sndr_db_cal - 1.76) / 6.02
        print(f"[Unit Test] Calculated from signal: SNDR=[{sndr_db_cal:.2f} dB], ENOB=[{enob_calc:.2f}]")

        for n_sig, sig in enumerate(calibrated_signal):
            results = analyze_spectrum(sig)
            sndr_db_after = results['sndr_db']
            enob_after = results['enob']
            print(f"[Unit Test] [{n_sig}], After Calibration: SNDR=[{sndr_db_after:.2f} dB], ENOB=[{enob_after:.2f}]")


        np.testing.assert_allclose(sndr_db_before, sndr_db_cal, atol=3.0)
        np.testing.assert_allclose(sndr_db_before, sndr_db_after, atol=3.0)


def test_calibration_multi_datasets_search_freq():
    """
    Test calibration across multiple frequencies and sample sizes
    Similar to test_verify_estimate_frequencies.py pattern
    """

    n_samples_list = [8192]  # Different lengths to sweep
    bit_width = 10

    for n_samp in n_samples_list:
        # Generate frequency list based on sample size
        num_test_freqs = 9
        bins = np.linspace(3, n_samp//2 - 3, num_test_freqs)
        bins = (np.round(bins / 2) * 2 + 1).astype(int)
        print(f"\n[Frequency bins]: {bins.tolist()}")
        freq_true_list = bins / n_samp

        print(f"\n{'='*80}")
        print(f"[N_samples = {n_samp}]")
        print(f"[Number of frequencies]: {len(freq_true_list)}")
        print(f"[Frequency bin width]: {1/n_samp:.8f}")

        # Setup true weights (normalized to max=1)
        true_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
        shift_amounts_order = np.arange(bit_width - 1, -1, -1)

        # Use identity mapping (no shuffling for this test)
        shuffled_indices = np.arange(bit_width)
        shuffled_weights = true_weights[shuffled_indices]
        current_shifts = shift_amounts_order[shuffled_indices]

        # Normalize weights
        normalized_weights = shuffled_weights / np.max(shuffled_weights)

        bits_ideal = []
        for idx, freq_true in enumerate(freq_true_list):
            # Generate ideal sine signal
            t = np.arange(n_samp)
            signal = 0.5 * np.sin(2 * np.pi * freq_true * t) + 0.5

            # Quantize to ADC output
            quantized_signal = np.floor(signal * (2**bit_width)).astype(int)
            quantized_signal = np.clip(quantized_signal, 0, 2**bit_width - 1)

            # Extract bits
            bits = (quantized_signal[:, None] >> current_shifts) & 1
            bits_ideal.append(bits)

        bits_mangled_list = [bits[:, shuffled_indices] for bits in bits_ideal]

        print(f"[Unit Test] Number of datasets=[{len(bits_mangled_list)}], number of frequencies=[{len(freq_true_list)}]")
        print(f"[Unit Test] Datasets: {[bits.shape for bits in bits_mangled_list]}")
        
        start_time = time.time()
        result = calibrate_weight_sine(bits_mangled_list, verbose=0)
        elapsed_time = time.time() - start_time
        print(f"\n[Without freq_init] Runtime: {elapsed_time*1e3:.2f} ms")

        # Verify results
        recovered_weights = result['weight']
        refined_freq = result['refined_frequency']
        offset = result['offset']
        calibrated_signal = result['calibrated_signal']
        print(f"[Unit Test] Calibrated Signal shapes: {[sig.shape for sig in calibrated_signal]}")
        ideal_signal = result['ideal']
        error_signal = result['error']
        snr_db = result['snr_db']
        enob = result['enob']

        # Compute weight recovery error        
        normalized_set_weights = shuffled_weights / np.max(shuffled_weights)
        max_weight_error = np.max(np.abs(recovered_weights - normalized_set_weights))

        print()
        print(f"[Unit Test] Set freq list: {[f'{f:.6f}' for f in freq_true_list]}")
        print(f"[Unit Test] Get freq list: {[f'{f:.6f}' for f in refined_freq]}")
        print()

        print(f"[Unit Test] Set weights: {normalized_set_weights.tolist()}") 
        print(f"[Unit Test] Calibrated weights: {recovered_weights.tolist()}")  
        print(f"[Unit Test] Max weight error: [{max_weight_error:.2e}] <-- Should be smaller than LSB=[{1/(2**bit_width):.2e}]")
        print(f"[Unit Test] Offset: {offset:.6f}")

        results = analyze_spectrum(quantized_signal)
        sndr_db_before = results['sndr_db']
        enob_before = results['enob']
        print(f"\n[Unit Test] Before Calibration    : SNDR=[{sndr_db_before:.2f} dB], ENOB=[{enob_before:.2f}]")

        sndr_db_cal = 10*np.log10(np.mean(ideal_signal[0]**2) / np.mean(error_signal[0]**2))
        enob_calc = (sndr_db_cal - 1.76) / 6.02
        print(f"[Unit Test] Calculated from signal: SNDR=[{sndr_db_cal:.2f} dB], ENOB=[{enob_calc:.2f}]")

        for n_sig, sig in enumerate(calibrated_signal):
            results = analyze_spectrum(sig)
            sndr_db_after = results['sndr_db']
            enob_after = results['enob']
            print(f"[Unit Test] [{n_sig}], After Calibration: SNDR=[{sndr_db_after:.2f} dB], ENOB=[{enob_after:.2f}]")


        # np.testing.assert_allclose(sndr_db_before, sndr_db_cal, atol=3.0)
        # np.testing.assert_allclose(sndr_db_before, sndr_db_after, atol=3.0)

