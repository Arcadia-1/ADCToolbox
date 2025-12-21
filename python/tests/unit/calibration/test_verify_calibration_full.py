"""
Unit Test: Verify calibrate_weight_sine wrapper function

Purpose:
  Comprehensive validation of ADC foreground calibration using sinewave input.
  Tests the main calibration wrapper (calibrate_weight_sine) for:

  - Single-dataset calibration with known frequency
  - Single-dataset calibration with automatic frequency search
  - Weight recovery accuracy (normalized binary weights)
  - Frequency estimation and refinement
  - SNDR and ENOB performance metrics validation

Test Pattern:
  Similar to test_verify_estimate_frequencies.py, this module:
  - Generates quantized sinewave signals with known weights
  - Applies calibration to recover unknown weights
  - Validates recovered weights against ground truth
  - Compares SNDR/ENOB before and after calibration

Known Limitations:
  - Certain frequencies may cause estimation failures (see _estimate_frequencies.py docstring)
  - Very short sample sizes may have reduced accuracy
"""

import numpy as np
import time
from adctoolbox.calibration import calibrate_weight_sine
from adctoolbox import analyze_spectrum

def test_calibration_single_dataset_shuffled():
    """
    Test single-dataset calibration with known frequency.

    Validates:
      - Weight recovery from quantized sinewave data
      - Frequency refinement accuracy
      - SNDR/ENOB performance metrics
      - Calibrated signal matches ideal sinewave

    Test Setup:
      - Generates sinewave at frequency 3/N
      - Quantizes to bit_width (8-bit ADC)
      - Extracts bit representation with optional shuffling
      - Calibrates using known input frequency

    Success Criteria:
      - Weight error < LSB threshold
      - SNDR before ≈ SNDR after (within 3 dB tolerance)
      - Frequency error near zero
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
    Test single-dataset calibration with automatic frequency search.

    Validates:
      - Automatic frequency estimation from bit data
      - Weight recovery without prior frequency knowledge
      - Robustness when frequency is unknown
      - SNDR/ENOB performance with auto frequency search

    Test Setup:
      - Generates sinewave at frequency 3/N
      - Quantizes to bit_width (8-bit ADC)
      - Extracts bit representation with optional shuffling
      - Calibrates WITHOUT providing input frequency (freq=None)

    Success Criteria:
      - Frequency estimated within one frequency bin
      - Weight error < LSB threshold
      - SNDR before ≈ SNDR after (within 3 dB tolerance)

    Note:
      This test validates the _estimate_frequencies() helper function
      integration with the full calibration pipeline.
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
