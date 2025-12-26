"""
ADC Signal Generator with Applier Pattern for Non-Idealities

This module provides the ADC_Signal_Generator class, which implements the Applier Pattern
for applying various ADC non-idealities to a base sine wave signal.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import CubicSpline

class ADC_Signal_Generator:
    """
    Generates ADC test signals with various non-idealities using the Applier Pattern.

    Each method accepts optional input_signal parameter. If None, uses clean base signal.
    Methods return signal with non-ideality applied. Supports chaining multiple effects.

    Parameters
    ----------
    N : int
        Number of samples
    Fs : float
        Sampling frequency (Hz)
    Fin : float
        Input signal frequency (Hz)
    A : float
        Input signal amplitude (e.g., 0.49)
    DC : float
        Input signal DC offset (e.g., 0.5)
    """

    def __init__(self, N, Fs, Fin, A, DC):
        """Initialize ADC_Signal_Generator with signal parameters."""
        self.N = N
        self.Fs = Fs
        self.Fin = Fin
        self.A = A
        self.DC = DC
        self.t = np.arange(N) / Fs

    def _get_base_signal(self):
        """Generate clean sine wave: A*sin(2π*Fin*t) + DC (no noise)."""
        return self.A * np.sin(2 * np.pi * self.Fin * self.t) + self.DC

    def _resolve_signal(self, input_signal):
        """Helper: return copy of input_signal, or generate base signal if None."""
        if input_signal is None:
            return self._get_base_signal()
        return input_signal.copy()

    def get_clean_signal(self):
        """Return clean sine wave: A*sin(2π*Fin*t) + DC (explicit public interface)."""
        return self.A * np.sin(2 * np.pi * self.Fin * self.t) + self.DC

    def apply_thermal_noise(self, input_signal=None, noise_rms=50e-6):
        """Apply white thermal noise. Params: input_signal (None->clean), noise_rms (default 50e-6)."""
        signal = self._resolve_signal(input_signal)
        noise = np.random.randn(self.N) * noise_rms
        return signal + noise

    def apply_quantization_noise(self, input_signal=None, n_bits=10, quant_range=(0.0, 1.0)):
        """ Apply quantization noise. Params: quant_range: tuple (v_min, v_max), e.g., (0, 1) or (-0.5, 0.5)."""
        signal = self._resolve_signal(input_signal)
        v_min, v_max = quant_range        
        # 1. Calculate LSB size based on Full Scale Range
        lsb = (v_max - v_min) / (2 ** n_bits)

        # 2. Voltage to Code (floor operation)
        codes = np.floor((signal - v_min) / lsb)

        # 3. Simulate ADC Saturation (Clip to valid code range 0 ~ 2^N-1)
        max_code = (2 ** n_bits) - 1
        codes = np.clip(codes, 0, max_code)

        # 4. Code to Voltage (Reconstruction)
        return codes * lsb + v_min
    
    def apply_jitter(self, input_signal=None, jitter_rms=2e-12):
        """
        Apply sampling jitter.
        
        Logic:
        - Case 1 (Source Generation): If input_signal is None, regenerates from scratch (Perfect Precision).
        - Case 2 (Chain Processing): If input is provided, uses Cubic Spline interpolation (Preserves previous errors).
        """
        # 1. Generate timing jitter
        t_jitter = np.random.randn(self.N) * jitter_rms

        if input_signal is None:
            # Case 1: Perfect mathematical generation (No interpolation error)
            total_phase = 2 * np.pi * self.Fin * (self.t + t_jitter)
            return self.A * np.sin(total_phase) + self.DC
        else:
            # Case 2: High-precision interpolation (Preserves previous distortions) 
            signal = input_signal.copy()
            t_sample = self.t + t_jitter
            # Cubic Spline is much better than linear interp for preserving SNR
            cs = CubicSpline(self.t, signal)
            return cs(t_sample)

    def apply_static_nonlinearity(self, input_signal=None, k2=0, k3=0, k4=0, k5=0):
        """Applies static nonlinear distortion to the signal using direct polynomial coefficients (k2 to k5)."""
        signal = self._resolve_signal(input_signal)
        signal_ac = signal - self.DC
        
        # Polynomial calculation (Vectorized)
        # y = x + k2*x^2 + ...
        # Optimized to avoid creating too many temporary arrays if coeffs are 0
        distortion = np.zeros_like(signal_ac)
        
        if k2 != 0: distortion += k2 * (signal_ac ** 2)
        if k3 != 0: distortion += k3 * (signal_ac ** 3)
        if k4 != 0: distortion += k4 * (signal_ac ** 4)
        if k5 != 0: distortion += k5 * (signal_ac ** 5)
        
        # Add distortion to original AC signal
        signal_distorted = signal_ac + distortion
        
        return signal_distorted + self.DC

    def apply_static_nonlinearity_hd(self, input_signal=None, hd2_dB=None, hd3_dB=None, hd4_dB=None, hd5_dB=None):
        """Converts specified Harmonic Distortion levels (dBc) to polynomial coefficients and applies the nonlinearity."""
        def db_to_k(db_val, order):
            if db_val is None: 
                return 0.0
                        
            amp_ratio = 10 ** (db_val / 20.0) # k = (2^(n-1) * 10^(dB/20)) / A^(n-1)
            return (2**(order-1) * amp_ratio) / (self.A**(order-1))
        
        _k2 = db_to_k(hd2_dB, 2)
        _k3 = db_to_k(hd3_dB, 3)
        _k4 = db_to_k(hd4_dB, 4)
        _k5 = db_to_k(hd5_dB, 5)
        
        return self.apply_static_nonlinearity(input_signal, k2=_k2, k3=_k3, k4=_k4, k5=_k5)
    
    def apply_memory_effect(self, input_signal=None, memory_strength=0.009):
        """Apply memory effect (charge injection). The previous MSB decision leaks back to the input.
        Params: input_signal, memory_strength (default 0.009)."""
        signal = self._resolve_signal(input_signal)
        # Quantize into MSB (4-bit coarse) and LSB (12-bit fine) stages
        msb = np.floor(signal * 2**4) / 2**4
        lsb = np.floor((signal - msb) * 2**12) / 2**12

        # Apply memory effect from previous MSB
        msb_shifted = np.roll(msb, shift=1)

        return msb + lsb + memory_strength * msb_shifted

    def apply_incomplete_sampling(self, input_signal=None, T_track=None, tau_nom=40e-12, coeff_k=0.15):
        """Apply dynamic nonlinearity (tracking/settling). Models slew-rate and signal-dependent settling errors. 
        Params: input_signal, T_track, tau_nom (default 40ps), coeff_k (default 0.15)."""
        signal = self._resolve_signal(input_signal)
        if T_track is None:
            T_track = (1 / self.Fs) * 0.2

        signal_ac = signal - self.DC
        vout = np.zeros(self.N)
        v_prev = 0
        for n in range(self.N):
            v_target = signal_ac[n]
            # Dynamic time constant changes with voltage (creates HD3)
            tau_dynamic = tau_nom * (1 + coeff_k * v_target ** 2)
            # Incomplete settling: output depends on previous state
            vout[n] = v_target + (v_prev - v_target) * np.exp(-T_track / tau_dynamic)
            v_prev = vout[n]

        return vout + self.DC
    
    def apply_ra_gain_error(self, input_signal=None, relative_gain=0.99, msb_bits=4, lsb_bits=12):
        """Apply interstage gain error (2-stage pipeline ADC). Params: input_signal, relative_gain (default 0.99 = 1% error)."""
        signal = self._resolve_signal(input_signal)
        signal_ac = signal - self.DC

        # Quantize into MSB and LSB stages
        msb = np.floor(signal_ac * 2**msb_bits) / 2**msb_bits
        lsb = np.floor((signal_ac - msb) * 2**lsb_bits) / 2**lsb_bits

        # Apply gain error to MSB stage, combine with LSB
        return msb * relative_gain + lsb + self.DC
    
    def apply_ra_gain_error_dynamic(self, input_signal=None, relative_gain=1, coeff_3=0.15, msb_bits=4, lsb_bits=12):
        """Applies dynamic gain error to the interstage residue amplifier in a pipeline ADC. G[n] is non-linearly dependent on the previous residue output magnitude (V_prev_ac^3), modeling HD3 memory effects."""

        signal = self._resolve_signal(input_signal)
        signal_ac = signal - self.DC

        v_output_ac = np.zeros(self.N)  
        # Memory term initialization: AC component of the previous residue amp output        
        v_residue_out_prev_ac = 0.0
        
        for n in range(self.N):
            v_in_ac = signal_ac[n]
            
            # Coarse Quantization (V_DAC)
            v_msb_code = np.floor(v_in_ac * 2**msb_bits) / 2**msb_bits            
            v_lsb_code = np.floor((v_in_ac - v_msb_code) * 2**lsb_bits) / 2**lsb_bits

            G_dynamic = relative_gain + coeff_3 * (v_residue_out_prev_ac ** 2)
            v_output_ac[n] = v_msb_code * G_dynamic + v_lsb_code
            
            # Update Memory Term
            v_residue_out_prev_ac = v_output_ac[n]

        return v_output_ac + self.DC
    
    def apply_reference_error(self, input_signal=None, settling_tau=2.0, droop_strength=0.01):
            """
            Apply Reference Incomplete Settling error (Vref Memory Effect).
            
            This simulates the reference voltage dropping due to load current (kick) 
            and failing to recover fully before the next sample.
            
            Params:
                input_signal: The input signal.
                settling_tau: Recovery time constant in units of samples (e.g., 2.0). 
                            Larger = Slower recovery = Worse settling.
                droop_strength: How much Vref drops proportional to signal amplitude (0.01 = 1%).
            """
            signal = self._resolve_signal(input_signal)
            signal_ac = signal - self.DC

            # Calculate Vref droop using IIR filter to simulate exponential decay         
            current_kick = droop_strength * np.abs(signal_ac)            
            decay = np.exp(-1.0 / settling_tau)            
            from scipy.signal import lfilter
            vref_droop = lfilter([1], [1, -decay], current_kick)            
            signal_settled = signal * (1.0 - vref_droop)
            
            return signal_settled + self.DC

    def apply_am_noise(self, input_signal=None, strength=0.01):
            """
            Apply random AM noise (Multiplicative Thermal Noise).
            Params: 
                input_signal: The signal to modulate.
                am_noise_depth: The RMS level of the noise relative to signal amplitude (default 0.1).
            Note:
                There is NO frequency parameter here because white noise contains ALL frequencies.
            """
            signal = self._resolve_signal(input_signal)
            signal_ac = signal - self.DC            
            am_envelope = 1 + strength * np.random.normal(loc=0.0, scale=1.0, size=len(self.t))
            
            signal_am = signal_ac * am_envelope
            return signal_am + self.DC
    
    def apply_am_tone(self, input_signal=None, am_tone_freq=500e3, am_tone_depth=0.05):
        """Apply AM tone (coherent modulation). Params: input_signal, am_tone_freq (default 500kHz), am_tone_depth (default 0.05)."""
        signal = self._resolve_signal(input_signal)
        signal_ac = signal - self.DC
        am_tone_env = 1 + am_tone_depth * np.sin(2 * np.pi * am_tone_freq * self.t)
        signal_am = signal_ac * am_tone_env
        return signal_am + self.DC

    def apply_clipping(self, input_signal=None, percentile_clip=1.0):
        """Apply hard clipping based on signal's percentile (e.g., 1.0 clips top/bottom 1%)."""
        
        signal = self._resolve_signal(input_signal) # Get base signal or input signal
        
        # 1. Determine the clipping thresholds dynamically using percentiles.
        lower_threshold = np.percentile(signal, percentile_clip)
        upper_threshold = np.percentile(signal, 100.0 - percentile_clip)
        
        # 2. Apply clipping: constrain signal values between the determined thresholds.
        signal_clipped = np.clip(signal, lower_threshold, upper_threshold)
        
        return signal_clipped

    def apply_drift(self, input_signal=None, drift_scale=5e-5):
        """Apply drift (low-frequency random walk). Params: input_signal, drift_scale (default 5e-5)."""
        signal = self._resolve_signal(input_signal)
        drift_steps = np.random.randn(self.N) * drift_scale
        drift_walk = np.cumsum(drift_steps)
        b, a = scipy_signal.butter(2, 0.001)
        drift = scipy_signal.filtfilt(b, a, drift_walk)
        return signal + drift

    def apply_glitch(self, input_signal=None, glitch_prob=0.00015, glitch_amplitude=0.1):
        """Apply random glitches. Params: input_signal, glitch_prob (default 0.00015 = 0.015%), glitch_amplitude (default 0.1)."""
        signal = self._resolve_signal(input_signal)
        glitch_mask = np.random.rand(self.N) < glitch_prob
        glitch = glitch_mask * glitch_amplitude
        return signal + glitch

    

