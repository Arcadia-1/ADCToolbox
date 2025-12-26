"""Standard ADC test configuration for non-ideality analysis examples.

This module defines a complete standard test setup including signal parameters and
15 ADC non-ideality cases, ensuring consistency across all analysis examples (exp_a20-a24).
"""

from adctoolbox import find_coherent_frequency
from adctoolbox.siggen import ADC_Signal_Generator

def get_batch_test_setup(hd2_target_dB=-80, hd3_target_dB=-70):
    """
    Get batch ADC test configuration with 15 non-ideality cases.

    This function creates a complete test setup with standardized parameters
    used across all analysis examples for consistent comparison.

    Parameters
    ----------
    hd2_target_dB : float, default=-80
        Target 2nd harmonic level in dBc
    hd3_target_dB : float, default=-70
        Target 3rd harmonic level in dBc

    Returns
    -------
    gen : ADC_Signal_Generator
        Configured signal generator instance
    cases : list of dict
        List of 15 test cases, each dict has:
        - 'title': str, descriptive name
        - 'func': callable, lambda that generates the signal
    params : dict
        Test parameters (N, Fs, Fin, A, DC, etc.) for reference

    Examples
    --------
    >>> from nonideality_cases import get_batch_test_setup
    >>> gen, cases, params = get_batch_test_setup()
    >>> signal = cases[0]['func']()  # Generate thermal noise case
    >>> print(f"Using Fs={params['Fs']/1e6:.0f} MHz")
    """

    # Standard signal parameters
    N = 2**16
    Fs = 800e6
    Fin_target = 97e6
    Fin, J = find_coherent_frequency(Fs, Fin_target, N)
    A = 0.49
    DC = 0.5
    base_noise = 50e-6

    # Create generator
    gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC)

    # Calculate coefficients for target harmonic levels
    hd2_amp = 10**(hd2_target_dB/20)
    hd3_amp = 10**(hd3_target_dB/20)
    k2_mag = (2 * hd2_amp) / A
    k3_mag = (4 * hd3_amp) / (A**2)

    # Define 15 standard non-ideality cases
    # All cases (except thermal noise) apply 10uV thermal noise as the LAST step
    cases = [
        {
            'title': 'Thermal Noise',
            'func': lambda: gen.apply_thermal_noise(noise_rms=180e-6)
        },
        {
            'title': 'Quantization Noise',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_quantization_noise(None, n_bits=10, quant_range=(0, 1)),
                noise_rms=10e-6
            )
        },
        {
            'title': 'Jitter Noise',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_jitter(None, jitter_rms=2e-12),
                noise_rms=10e-6
            )
        },
        {
            'title': 'AM Noise',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_am_noise(None, strength=0.0005),
                noise_rms=10e-6
            )
        },
        {
            'title': f'Static HD2 ({hd2_target_dB} dBc)',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_static_nonlinearity(None, k2=k2_mag, k3=0),
                noise_rms=10e-6
            )
        },
        {
            'title': f'Static HD3 ({hd3_target_dB} dBc)',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_static_nonlinearity(None, k2=0, k3=k3_mag),
                noise_rms=10e-6
            )
        },
        {
            'title': 'Memory Effect',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_memory_effect(None, memory_strength=0.009),
                noise_rms=10e-6
            )
        },
        {
            'title': 'Incomplete Settling',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_incomplete_sampling(None, T_track=0.2/Fs, coeff_k=0.09),
                noise_rms=10e-6
            )
        },
        {
            'title': 'RA Gain Error',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_ra_gain_error(None, msb_bits=4, lsb_bits=8, relative_gain=0.99),
                noise_rms=10e-6
            )
        },
        {
            'title': 'RA Dynamic Gain',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_ra_gain_error_dynamic(None, msb_bits=4, lsb_bits=8, coeff_3=0.15),
                noise_rms=10e-6
            )
        },
        {
            'title': 'AM Tone',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_am_tone(None, am_tone_freq=500e3, am_tone_depth=0.05),
                noise_rms=10e-6
            )
        },
        {
            'title': 'Clipping',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_clipping(None, percentile_clip=1.0),
                noise_rms=10e-6
            )
        },
        {
            'title': 'Drift',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_drift(None, drift_scale=5e-5),
                noise_rms=10e-6
            )
        },
        {
            'title': 'Reference Error',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_reference_error(None, settling_tau=0.1, droop_strength=0.002),
                noise_rms=10e-6
            )
        },
        {
            'title': 'Glitch',
            'func': lambda: gen.apply_thermal_noise(
                gen.apply_glitch(None, glitch_prob=0.002, glitch_amplitude=0.1),
                noise_rms=10e-6
            )
        },
    ]

    # Parameters dictionary for reference
    params = {
        'N': N,
        'Fs': Fs,
        'Fin': Fin,
        'Fin_bin': J,
        'A': A,
        'DC': DC,
        'base_noise': base_noise,
        'B': 12,  # ADC resolution in bits
        'adc_range': [0, 1],  # ADC full-scale range
    }

    return gen, cases, params
