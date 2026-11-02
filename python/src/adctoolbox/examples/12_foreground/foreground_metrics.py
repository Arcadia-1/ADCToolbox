"""Coherent spectral and absolute-error ratios for foreground examples."""
import numpy as np


def metrics(prediction, clean, tone_bins):
    # Known bins are used ONLY by the evaluator. For one tone this is SNDR /
    # SFDR; for multiple tones use total desired-tone power / noise+distortion
    # and strongest desired tone / largest unwanted bin, respectively.
    power = abs(np.fft.rfft(prediction-prediction.mean()))**2
    # One-sided Parseval weights; DC is excluded and Nyquist is not doubled.
    power[1:-1] *= 2
    desired = float(power[tone_bins].sum())
    strongest = float(power[tone_bins].max())
    power[tone_bins] = 0.
    power[0] = 0.
    return {"spectral_ratio_db": float(10*np.log10(desired/max(power.sum(),1e-300))),
            "spur_ratio_db": float(10*np.log10(strongest/max(power.max(),1e-300))),
            "error_snr_db": float(10*np.log10(np.var(clean)/max(np.mean((prediction-clean)**2),1e-300)))}
