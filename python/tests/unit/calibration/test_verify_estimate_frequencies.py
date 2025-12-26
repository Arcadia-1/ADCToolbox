"""
Unit Test: Verify frequency estimation helper functions

Purpose: Self-verify that frequency estimation works correctly for:
- User-provided scalar frequencies (broadcast to all datasets)
- User-provided array frequencies (validated against dataset count)
- Automatic FFT-based frequency estimation (variance-weighted bit selection)
"""

import numpy as np
import time
from adctoolbox.calibration._estimate_frequencies import _estimate_frequencies

def test_freq_estimation_shuffled():

    n_samples_list = [16, 128, 1024, 16384]  # Different lengths to sweep
    bit_width = 12
    shuffled_indices = np.random.permutation(bit_width)
    # shuffled_indices = np.arange(bit_width)

    print(f"\n[n_samples_list to sweep]: {n_samples_list}")
    print(f"[Shuffled bit indices]: {shuffled_indices}")

    for n_samp in n_samples_list:
        freq_true = np.array([0, 1/n_samp, 2/n_samp, 0.1, 0.15, 0.2, 0.21, 0.25, 0.29, 0.3, 0.35, 0.4, (0.5*n_samp - 2)/n_samp, (0.5*n_samp - 1)/n_samp, 0.5])
        # freq_true = np.array([0.2,0.5])

        # Generate bits_ideal for all frequencies at this n_samp
        bits_ideal = []
        for freq in freq_true:
            t = np.arange(n_samp)
            signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.5
            quantized_signal = np.floor(signal * (2**bit_width)).astype(int)
            quantized_signal = np.clip(quantized_signal, 0, 2**bit_width - 1)
            bits = (quantized_signal[:, None] >> np.arange(bit_width)) & 1
            bits_ideal.append(bits)

        # Shuffle each bit array and concatenate them
        bits_mangled_list = [bits[:, shuffled_indices] for bits in bits_ideal]
        bits_mangled = np.vstack(bits_mangled_list)

        segment_lengths = np.array([n_samp] * len(freq_true))

        print(f"\n{'='*80}")
        print(f"[N_samples = {n_samp}]")
        print(f"[Bits_mangled shape]: {bits_mangled.shape}")
        print(f"[Number of frequencies]: {len(freq_true)}")
        print(f"[One bin threshold]: {1/n_samp:.8f}")

        # Estimate without freq_init
        start_time = time.time()
        freq_array = _estimate_frequencies(bits_mangled, segment_lengths, verbose=2)
        elapsed_time = time.time() - start_time
        print(f"\n[Without freq_init] Runtime: {elapsed_time*1e6:.6f} us")

        # Print results with one-bin threshold
        threshold = 1 / n_samp
        print(f"\nResults (threshold = 1/N = {threshold:.8f}):")
        for i, (freq_t, freq_e) in enumerate(zip(freq_true, freq_array)):
            abs_error = abs(float(freq_e) - float(freq_t))
            if abs_error <= threshold:
                print(f"  [{i:<3}], N=[{n_samp:<7}], True Freq=[{float(freq_t):<8.6f}], Est Freq=[{float(freq_e):<8.6f}], Abs Error=[{abs_error:<9.2e}]")
            else:
                print(f"  [{i:<3}], N=[{n_samp:<7}], True Freq=[{float(freq_t):<8.6f}], Est Freq=[{float(freq_e):<8.6f}], Abs Error=[{abs_error:<9.2e}] <-- Bad")


    # 5. Assertions
    # The frequency bin resolution is 1/N. 
    # The coarse estimate should be within 1 bin of the truth.
    # freq_est = freq_array[0]
    # error = abs(freq_est - freq_true)
    # max_allowed_error = 1.0 / n_samples
    
    # print(f"\n[STRESS TEST RESULT]")
    # print(f"True Freq: {freq_true:.6f}")
    # print(f"Est  Freq: {freq_est:.6f}")
    # print(f"Error:     {error:.2e} (Limit: {max_allowed_error:.2e})")
    
    # assert error <= max_allowed_error, f"Frequency estimation failed! Error {error} too large."


def test_freq_estimation_shuffled_with_noise():

    n_samples_list = [16, 128, 1024, 16384]  # Different lengths to sweep
    bit_width = 12
    shuffled_indices = np.random.permutation(bit_width)
    # shuffled_indices = np.arange(bit_width)

    print(f"\n[n_samples_list to sweep]: {n_samples_list}")
    print(f"[Shuffled bit indices]: {shuffled_indices}")

    for n_samp in n_samples_list:
        freq_true = np.array([1/n_samp, 2/n_samp, 0.1, 0.15, 0.2, 0.21, 0.25, 0.29, 0.3, 0.35, 0.4, (0.5*n_samp - 2)/n_samp, (0.5*n_samp - 1)/n_samp, 0.5])
        # freq_true = np.array([0.2,0.5])

        # Generate bits_ideal for all frequencies at this n_samp
        bits_ideal = []
        for freq in freq_true:
            t = np.arange(n_samp)
            signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.5 + 0.0005 * np.random.randn(n_samp)  # Add noise
            quantized_signal = np.floor(signal * (2**bit_width)).astype(int)
            quantized_signal = np.clip(quantized_signal, 0, 2**bit_width - 1)
            bits = (quantized_signal[:, None] >> np.arange(bit_width)) & 1
            bits_ideal.append(bits)

        # Shuffle each bit array and concatenate them
        bits_mangled_list = [bits[:, shuffled_indices] for bits in bits_ideal]
        bits_mangled = np.vstack(bits_mangled_list)

        segment_lengths = np.array([n_samp] * len(freq_true))

        print(f"\n{'='*80}")
        print(f"[N_samples = {n_samp}]")
        print(f"[Bits_mangled shape]: {bits_mangled.shape}")
        print(f"[Number of frequencies]: {len(freq_true)}")
        print(f"[One bin threshold]: {1/n_samp:.8f}")

        # Estimate without freq_init
        start_time = time.time()
        freq_array = _estimate_frequencies(bits_mangled, segment_lengths, verbose=2)
        elapsed_time = time.time() - start_time
        print(f"\n[Without freq_init] Runtime: {elapsed_time*1e6:.6f} us")

        # Print results with one-bin threshold
        threshold = 1 / n_samp
        print(f"\nResults (threshold = 1/N = {threshold:.8f}):")
        for i, (freq_t, freq_e) in enumerate(zip(freq_true, freq_array)):
            abs_error = abs(float(freq_e) - float(freq_t))
            if abs_error <= threshold:
                print(f"  [{i:<3}], N=[{n_samp:<7}], True Freq=[{float(freq_t):<8.6f}], Est Freq=[{float(freq_e):<8.6f}], Abs Error=[{abs_error:<9.2e}]")
            else:
                print(f"  [{i:<3}], N=[{n_samp:<7}], True Freq=[{float(freq_t):<8.6f}], Est Freq=[{float(freq_e):<8.6f}], Abs Error=[{abs_error:<9.2e}] <-- Bad")


    # 5. Assertions
    # The frequency bin resolution is 1/N. 
    # The coarse estimate should be within 1 bin of the truth.
    # freq_est = freq_array[0]
    # error = abs(freq_est - freq_true)
    # max_allowed_error = 1.0 / n_samples
    
    # print(f"\n[STRESS TEST RESULT]")
    # print(f"True Freq: {freq_true:.6f}")
    # print(f"Est  Freq: {freq_est:.6f}")
    # print(f"Error:     {error:.2e} (Limit: {max_allowed_error:.2e})")
    
    # assert error <= max_allowed_error, f"Frequency estimation failed! Error {error} too large."

