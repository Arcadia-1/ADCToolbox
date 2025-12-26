# Example Print Outputs: 07_calculate_metric

This document records the console output from all examples in `python/src/adctoolbox/examples/07_calculate_metric/`.

---

## exp_b01_aliasing_nyquist_zones.py

**Description**: Demonstrate aliasing and Nyquist zone calculations.

```
[Aliasing] Fs = 1100.0 MHz, Fin_target = 123.0 MHz -> F_aliased = 123.0 MHz
[Aliasing 500 frequencies] [Input = 0.0 - 3300.0 MHz] [Output = 0.00 - 548.90 MHz]

[Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\07_calculate_metric\output\exp_b05_aliasing.png]
```

---

## exp_b02_unit_conversions.py

**Description**: Unit conversion utilities.

**Status**: Not functional - missing dependencies

```
ImportError: cannot import name 'lsb_to_volts' from 'adctoolbox'
```

---

## exp_b03_calculate_fom.py

**Description**: Calculate ADC figures of merit.

**Status**: Not functional - missing dependencies

```
ModuleNotFoundError: No module named 'adctoolbox.common'
```

---

## exp_b05_amplitudes_to_snr.py

**Description**: Convert signal/noise amplitudes to SNR metrics.

```
[Figure saved] -> D:\ADCToolbox\python\src\adctoolbox\examples\07_calculate_metric\output\exp_b05_snr_calculations.png

======================================================================
Summary: SNR = 20*log10(A_RMS / noise_RMS) = 20*log10(A/sqrt(2) / sigma)
======================================================================
Signal Amplitude: A = 0.5 V, FSR = 1.0 V

ADC Quantization Noise:
   6-bit: Q-noise= 4510.5 uV, SNR=37.88 dB (Theory=37.88 dB)
   8-bit: Q-noise= 1127.6 uV, SNR=49.93 dB (Theory=49.92 dB)
  10-bit: Q-noise=  281.9 uV, SNR=61.97 dB (Theory=61.96 dB)
  12-bit: Q-noise=   70.5 uV, SNR=74.01 dB (Theory=74.00 dB)
  14-bit: Q-noise=   17.6 uV, SNR=86.05 dB (Theory=86.04 dB)
======================================================================
```

---

## exp_b06_convert_nsd_snr.py

**Description**: Convert between NSD and SNR metrics.

```
[Figure saved] -> D:\ADCToolbox\python\src\adctoolbox\examples\07_calculate_metric\output\exp_b06_nsd_snr_conversions.png

[SNR -> NSD -> SNR Round-trip]
  [SNR = 85.30 dB] -> [NSD = -121.22 dBFS/Hz] -> [SNR = 85.30 dB]
```

---

## Summary

All examples in `07_calculate_metric` demonstrate metric calculation utilities:

**Total Examples**: 5 (3 functional, 2 not functional)

**Functional Examples**:
- **exp_b01**: Aliasing and Nyquist zone calculations
- **exp_b05**: Amplitude to SNR conversions
- **exp_b06**: NSD to SNR conversions

**Not Functional** (missing dependencies):
- **exp_b02**: Unit conversions (missing `lsb_to_volts`)
- **exp_b03**: FOM calculations (missing `adctoolbox.common` module)

**Key Features**:
- Frequency aliasing calculations
- SNR metric conversions
- NSD (Noise Spectral Density) calculations
- Quantization noise analysis
