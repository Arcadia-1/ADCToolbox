"""
Jitter calculation for ADC analysis.

Based on the MATLAB golden reference: matlab_reference/test_jitter.m
"""

import numpy as np



def calculate_jitter(data, fin, Fin_Hz):
    """
    Calculate jitter from ADC data using error histogram analysis.

    Algorithm (from MATLAB reference test_jitter.m lines 43-53):
    1. [emean, erms, phase_code] = errHistSine(data, 99, J/N, 1, "disp", 0);
    2. amp_err = sqrt(erms.^2 - min(erms)^2);
    3. norm_amp_err = amp_err * 2 / (max(data) - min(data));
    4. phase_slope = abs(sin(phase_code/360*2*pi));
    5. jitter_on_phase = norm_amp_err ./ phase_slope / (2*pi*Fin);
    6. jitter_on_phase(~isfinite(jitter_on_phase)) = 0;
    7. jitter_rms = mean(jitter_on_phase(2:end));

    Args:
        data: ADC output data (numpy array)
        fin: Normalized frequency (0 to 0.5)
        Fin_Hz: Actual frequency in Hz

    Returns:
        jitter_rms: RMS jitter in seconds
    """
    from .errHistSine import errHistPhase

    data = np.asarray(data).flatten()

    # Step 1: Get error histogram in phase domain (bin_count=99 to match MATLAB)
    emean, erms, phase_code, _ = errHistPhase(data, bin_count=99, fin=fin, disp=0)

    # Step 2: Calculate amplitude error (remove minimum baseline)
    amp_err = np.sqrt(erms**2 - np.min(erms)**2)

    # Step 3: Normalize by signal range
    norm_amp_err = amp_err * 2 / (np.max(data) - np.min(data))

    # Step 4: Calculate phase slope (derivative of sine at each phase)
    phase_slope = np.abs(np.sin(phase_code / 360 * 2 * np.pi))

    # Step 5: Calculate jitter at each phase bin
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        jitter_on_phase = norm_amp_err / phase_slope / (2 * np.pi * Fin_Hz)

    # Step 6: Remove non-finite values (inf, nan)
    jitter_on_phase[~np.isfinite(jitter_on_phase)] = 0

    # Step 7: Return mean jitter (excluding first bin to match MATLAB: mean(jitter_on_phase(2:end)))
    # MATLAB索引从1开始，jitter_on_phase(2:end)意味着从第二个元素到最后
    # Python索引从0开始，所以是[1:]
    jitter_rms = np.mean(jitter_on_phase[1:])

    return jitter_rms


if __name__ == "__main__":
    # Quick test
    N = 2**14
    Fs = 10e9  # 10 GHz
    Fin = 100e6  # 100 MHz

    # Find coherent bin
    from findBin import find_bin
    J = find_bin(Fs, Fin, N)
    fin_norm = J / N
    Fin_actual = J / N * Fs

    # Generate test signal with known jitter
    Tj_set = 1e-12  # 1 ps
    phase_noise_rms = 2 * np.pi * Fin_actual * Tj_set

    Ts = 1 / Fs
    theta = 2 * np.pi * Fin_actual * np.arange(N) * Ts
    phase_jitter = np.random.randn(N) * phase_noise_rms

    data = np.sin(theta + phase_jitter) * 0.49 + 0.5 + np.random.randn(N) * 0.00001

    # Calculate jitter
    jitter_calc = calculate_jitter(data, fin=fin_norm, Fin_Hz=Fin_actual)

    print(f"Set jitter: {Tj_set * 1e15:.2f} fs")
    print(f"Calculated jitter: {jitter_calc * 1e15:.2f} fs")
    print(f"Ratio: {jitter_calc / Tj_set:.2f}")
