Expected Output: 04_debug_analog
======================================

This document shows the expected console output and example figures from all examples in `python/src/adctoolbox/examples/04_debug_analog/`.

Summary
-------

All examples in `04_debug_analog` demonstrate analog output analysis capabilities:

**Total Examples**: 15

**Categories**:
- **Sine Fitting**: exp_a01 (4-parameter fitting)
- **Error Analysis**: exp_a02-a04 (by value, by phase, jitter calculation)
- **Harmonic Decomposition**: exp_a11-a12 (time domain and polar)
- **Statistical Analysis**: exp_a21-a25 (PDF, spectrum, autocorrelation, envelope, comparison)
- **Nonlinearity**: exp_a31-a32 (static nonlinearity fitting, INL/DNL)
- **Phase Plane**: exp_a41-a42 (standard and error phase plane)


---

exp_a01_fit_sine_4param.py
--------------------------

**Description**: Basic 4-parameter sine fitting with noise.

.. code-block:: none

   [Sinewave] [Fs=800.0 MHz] [Fin=10.123457 MHz] [Amplitude=0.499 V] [DC=0.500 V] [Noise RMS=20.00 mV]
   
   [Expected] [DC=  0.5000 V] [Amplitude=  0.4990 V] [Freq=10.1234567 MHz] [Phase=  0.0000 deg]
   [Fitted  ] [DC=  0.4999 V] [Amplitude=  0.4985 V] [Freq=10.1233928 MHz] [Phase=  0.1283 deg]
   
   [Residual error ] fitted rms=19.73 mV, input noise=20.00 mV, error percentage=1.37% -> [PASS]
   [Frequency error] freq error=63.9330 Hz, percentage=0.000632% -> [PASS]
   [Phase error    ] phase error=0.1283 deg -> [PASS]
   [Reconstruction ] reconstruction error (rms) =5.62e-04 V -> [PASS]
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a01_fit_sine_4param.png]

---

exp_a02_analyze_error_by_value.py
---------------------------------

**Description**: Analyze ADC errors binned by input signal level.

.. code-block:: none

   [Config] Fs=800 MHz, Fin=10.12 MHz, N=8192
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a02_analyze_error_by_value_bins.png]

---

exp_a03_analyze_error_by_phase.py
---------------------------------

**Description**: Decompose noise into AM/PM components vs signal phase.

.. code-block:: none

   [Config] Fs=800 MHz, Fin=10.12 MHz, N=65536, A=0.49
   
   ================================================================================
   Mode: include_base_noise=True
   ================================================================================
   
   === Figure 1: Pure Noise Cases (With Base Noise) ===
   Thermal Only
     [Expected  ] [AM=   0 uV] [PM=   0 uV] [Base=  50 uV] [Total=50.0 uV]
     [Calculated] [AM= 5.2 uV] [PM= 0.0 uV] [Base=49.7 uV] [Total=49.9 uV] [R2=0.996]
   
   AM Only
     [Expected  ] [AM=  50 uV] [PM=   0 uV] [Base=   0 uV] [Total=35.4 uV]
     [Calculated] [AM=50.1 uV] [PM= 0.0 uV] [Base= 0.0 uV] [Total=35.4 uV] [R2=0.991]
   
   PM Only
     [Expected  ] [AM=   0 uV] [PM=  50 uV] [Base=   0 uV] [Total=35.4 uV]
     [Calculated] [AM= 0.0 uV] [PM=50.3 uV] [Base= 0.0 uV] [Total=35.5 uV] [R2=0.991]
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a03_analyze_error_by_phase_1_with_base_noise.png]
   
   === Figure 2: Mixed Noise Cases (With Base Noise) ===
   AM + Thermal
     [Expected  ] [AM=  50 uV] [PM=   0 uV] [Base=  30 uV] [Total=46.4 uV]
     [Calculated] [AM=50.0 uV] [PM= 0.0 uV] [Base=30.0 uV] [Total=46.4 uV] [R2=0.980]
   
   PM + Thermal
     [Expected  ] [AM=   0 uV] [PM=  50 uV] [Base=  30 uV] [Total=46.4 uV]
     [Calculated] [AM= 0.0 uV] [PM=50.0 uV] [Base=30.0 uV] [Total=46.4 uV] [R2=0.985]
   
   AM + PM + Thermal
     [Expected  ] [AM=   0 uV] [PM=   0 uV] [Base=  58 uV] [Total=58.0 uV]
     [Calculated] [AM= 0.0 uV] [PM= 7.9 uV] [Base=57.8 uV] [Total=58.1 uV] [R2=0.997]
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a03_analyze_error_by_phase_2_with_base_noise.png]
   
   === Figure 3: Mixed Noise Cases (AM+PM) (With Base Noise) ===
   AM + PM (equal)
     [Expected  ] [AM=   0 uV] [PM=   0 uV] [Base=  50 uV] [Total=50.0 uV]
     [Calculated] [AM= 0.0 uV] [PM= 2.8 uV] [Base=50.0 uV] [Total=50.1 uV] [R2=0.997]
   
   AM(30) + PM(50)
     [Expected  ] [AM=   0 uV] [PM=  40 uV] [Base=  30 uV] [Total=41.2 uV]
     [Calculated] [AM= 0.0 uV] [PM=40.3 uV] [Base=29.8 uV] [Total=41.2 uV] [R2=0.974]
   
   AM(50) + PM(30)
     [Expected  ] [AM=  40 uV] [PM=   0 uV] [Base=  30 uV] [Total=41.2 uV]
     [Calculated] [AM=39.6 uV] [PM= 0.0 uV] [Base=30.1 uV] [Total=41.1 uV] [R2=0.962]
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a03_analyze_error_by_phase_3_with_base_noise.png]
   
   ================================================================================
   Mode: include_base_noise=False
   ================================================================================
   
   === Figure 1: Pure Noise Cases (Without Base Noise) ===
   Thermal Only
     [Expected  ] [AM=   0 uV] [PM=   0 uV] [Base=  50 uV] [Total=50.0 uV]
     [Calculated] [AM=50.0 uV] [PM=49.7 uV] [Base= 0.0 uV] [Total=49.9 uV] [R2=0.996]
   
   AM Only
     [Expected  ] [AM=  50 uV] [PM=   0 uV] [Base=   0 uV] [Total=35.4 uV]
     [Calculated] [AM=50.1 uV] [PM= 0.0 uV] [Base= 0.0 uV] [Total=35.4 uV] [R2=0.991]
   
   PM Only
     [Expected  ] [AM=   0 uV] [PM=  50 uV] [Base=   0 uV] [Total=35.4 uV]
     [Calculated] [AM= 0.0 uV] [PM=50.3 uV] [Base= 0.0 uV] [Total=35.5 uV] [R2=0.991]
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a03_analyze_error_by_phase_1_no_base_noise.png]
   
   === Figure 2: Mixed Noise Cases (Without Base Noise) ===
   AM + Thermal
     [Expected  ] [AM=  50 uV] [PM=   0 uV] [Base=  30 uV] [Total=46.4 uV]
     [Calculated] [AM=58.3 uV] [PM=30.0 uV] [Base= 0.0 uV] [Total=46.4 uV] [R2=0.980]
   
   PM + Thermal
     [Expected  ] [AM=   0 uV] [PM=  50 uV] [Base=  30 uV] [Total=46.4 uV]
     [Calculated] [AM=30.0 uV] [PM=58.3 uV] [Base= 0.0 uV] [Total=46.4 uV] [R2=0.985]
   
   AM + PM + Thermal
     [Expected  ] [AM=   0 uV] [PM=   0 uV] [Base=  58 uV] [Total=58.0 uV]
     [Calculated] [AM=57.8 uV] [PM=58.4 uV] [Base= 0.0 uV] [Total=58.1 uV] [R2=0.997]
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a03_analyze_error_by_phase_2_no_base_noise.png]
   
   === Figure 3: Mixed Noise Cases (AM+PM) (Without Base Noise) ===
   AM + PM (equal)
     [Expected  ] [AM=   0 uV] [PM=   0 uV] [Base=  50 uV] [Total=50.0 uV]
     [Calculated] [AM=50.0 uV] [PM=50.1 uV] [Base= 0.0 uV] [Total=50.1 uV] [R2=0.997]
   
   AM(30) + PM(50)
     [Expected  ] [AM=   0 uV] [PM=  40 uV] [Base=  30 uV] [Total=41.2 uV]
     [Calculated] [AM=29.8 uV] [PM=50.1 uV] [Base= 0.0 uV] [Total=41.2 uV] [R2=0.974]
   
   AM(50) + PM(30)
     [Expected  ] [AM=  40 uV] [PM=   0 uV] [Base=  30 uV] [Total=41.2 uV]
     [Calculated] [AM=49.8 uV] [PM=30.1 uV] [Base= 0.0 uV] [Total=41.1 uV] [R2=0.962]
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a03_analyze_error_by_phase_3_no_base_noise.png]

---

exp_a04_jitter_calculation.py
-----------------------------

**Description**: Calculate jitter from spectrum analysis.

.. code-block:: none

   [Config] Fs=7 GHz, N=65536, A=0.490 V, DC=0.000 V
   [Jitter Sweep] 1.0 fs to 10.0 ps (30 points)
   [Frequencies] Testing 3 frequencies: [100, 1000, 2000] MHz
   
   ================================================================================
   Row 1: Without Thermal Noise
   ================================================================================
   [1/3] Fin_target = 100 MHz, Fin = 100.082 MHz, Bin = 937/65536
     Correlation = 1.0000, Avg Error = 0.54%
   [2/3] Fin_target = 1000 MHz, Fin = 1000.076 MHz, Bin = 9363/65536
     Correlation = 1.0000, Avg Error = 0.52%
   [3/3] Fin_target = 2000 MHz, Fin = 2000.046 MHz, Bin = 18725/65536
     Correlation = 1.0000, Avg Error = 0.39%
   
   ================================================================================
   Row 2: With 50 uV Thermal Noise
   ================================================================================
   [1/3] Fin_target = 100 MHz, Fin = 100.082 MHz, Bin = 937/65536
     Correlation = 1.0000, Avg Error = 133.18%
   [2/3] Fin_target = 1000 MHz, Fin = 1000.076 MHz, Bin = 9363/65536
     Correlation = 1.0000, Avg Error = 5.23%
   [3/3] Fin_target = 2000 MHz, Fin = 2000.046 MHz, Bin = 18725/65536
     Correlation = 1.0000, Avg Error = 4.44%
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a04_jitter_calculation.png]

---

exp_a11_decompose_harmonics.py
------------------------------

**Description**: Decompose signal into harmonics in time domain.

.. code-block:: none

   [Config] Fs=800 MHz, Fin=10.12 MHz, N=8192
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a11_decompose_harmonics.png]

---

exp_a12_decompose_harmonics_polar.py
------------------------------------

**Description**: Decompose harmonics with polar phase visualization.

.. code-block:: none

   [Config] Fs=800 MHz, Fin=99.902344 MHz, Bin=1023, N=8192
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a12_decompose_harmonics_polar.png]

---

exp_a21_analyze_error_pdf.py
----------------------------

**Description**: Analyze error probability density functions for various non-idealities.

.. code-block:: none

   [Config] Fs=800 MHz, Fin=97.0 MHz, Bin=7947, N=65536
   [Config] A=0.490 V, DC=0.500 V, Resolution=12 bits
   
   ====================================================================================================
   Case | Non-Ideality                   | mu (LSB)   | sigma (LSB) | KL Div
   ----------------------------------------------------------------------------------------------------
   1    | Thermal Noise                  |     -0.000 |      0.754 |     0.0002
   2    | Quantization Noise             |      0.000 |      1.172 |     0.1231
   3    | Jitter Noise                   |     -0.000 |      1.762 |     0.0522
   4    | AM Noise                       |     -0.000 |      0.722 |     0.0494
   5    | Static HD2 (-80 dBc)           |      0.000 |      0.151 |     0.1316
   6    | Static HD3 (-70 dBc)           |     -0.000 |      0.459 |     0.2649
   7    | Memory Effect                  |      0.000 |      0.711 |     0.0271
   8    | Incomplete Settling            |      0.000 |      0.085 |     0.0354
   9    | RA Gain Error                  |     -0.000 |      4.756 |     0.0946
   10   | RA Dynamic Gain                |     -0.000 |     14.053 |     0.0787
   11   | AM Tone                        |     -0.000 |     48.786 |     0.0648
   12   | Clipping                       |     -0.000 |      0.111 |     0.7130
   13   | Drift                          |      0.000 |     21.047 |     0.2127
   14   | Reference Error                |     -0.000 |      0.678 |     0.2261
   15   | Glitch                         |     -0.000 |     15.241 |     1.6148
   ====================================================================================================
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a21_analyze_error_pdf.png]

---

exp_a22_analyze_error_spectrum.py
---------------------------------

**Description**: Analyze error spectrum for various non-idealities.

.. code-block:: none

   [Config] Fs=800 MHz, Fin=97.0 MHz, Bin=7947, N=65536
   [Config] A=0.490 V, DC=0.500 V
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a22_analyze_error_spectrum.png]

---

exp_a23_analyze_error_autocorrelation.py
----------------------------------------

**Description**: Analyze error autocorrelation for various non-idealities.

.. code-block:: none

   [Config] Fs=800 MHz, Fin=97.0 MHz, Bin=7947, N=65536
   [Config] A=0.490 V, DC=0.500 V
   
   ====================================================================================================
   Case | Non-Ideality                   | ACF[0]     | ACF[1]     | ACF[10]
   ----------------------------------------------------------------------------------------------------
   1    | Thermal Noise                  |     1.0000 |     0.0029 |     0.0016
   2    | Quantization Noise             |     1.0000 |     0.0249 |    -0.0024
   3    | Jitter Noise                   |     1.0000 |    -0.0020 |     0.0015
   4    | AM Noise                       |     1.0000 |     0.0001 |     0.0018
   5    | Static HD2 (-80 dBc)           |     1.0000 |     0.0435 |    -0.8228
   6    | Static HD3 (-70 dBc)           |     1.0000 |    -0.6501 |    -0.6424
   7    | Memory Effect                  |     1.0000 |    -0.0443 |    -0.1179
   8    | Incomplete Settling            |     1.0000 |    -0.4859 |    -0.4789
   9    | RA Gain Error                  |     1.0000 |     0.0286 |     0.0267
   10   | RA Dynamic Gain                |     1.0000 |    -0.5601 |    -0.5696
   11   | AM Tone                        |     1.0000 |     0.7235 |     0.2326
   12   | Clipping                       |     1.0000 |    -0.0193 |    -0.0082
   13   | Drift                          |     1.0000 |     0.9999 |     0.9994
   14   | Reference Error                |     1.0000 |    -0.0781 |    -0.7964
   15   | Glitch                         |     1.0000 |     0.0041 |    -0.0024
   ====================================================================================================
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a23_analyze_error_autocorrelation.png]

---

exp_a24_analyze_error_envelope_spectrum.py
------------------------------------------

**Description**: Analyze error envelope spectrum for various non-idealities.

.. code-block:: none

   [Config] Fs=800 MHz, Fin=97.0 MHz, Bin=7947, N=65536
   [Config] A=0.490 V, DC=0.500 V
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a24_analyze_error_envelope_spectrum.png]

---

exp_a25_spectra.py
------------------

**Description**: Compare spectra across various non-idealities.

.. code-block:: none

   [Config] Fs=800 MHz, Fin=97.0 MHz, Bin=7947, N=65536
   [Config] A=0.490 V, DC=0.500 V, ADC Range=[0, 1]
   
   ====================================================================================================
   Case | Non-Ideality                   | SFDR (dB)  | SNDR (dB)  | THD (dB)
   ----------------------------------------------------------------------------------------------------
   1    | Thermal Noise                  |      97.15 |      65.72 |     -99.31
   2    | Quantization Noise             |      83.93 |      61.84 |     -97.93
   3    | Jitter Noise                   |      89.85 |      58.34 |     -94.75
   4    | AM Noise                       |      97.86 |      65.96 |     -99.31
   5    | Static HD2 (-80 dBc)           |      80.01 |      79.66 |     -80.18
   6    | Static HD3 (-70 dBc)           |      70.01 |      69.97 |     -70.18
   7    | Memory Effect                  |      76.44 |      66.18 |     -79.37
   8    | Incomplete Settling            |      85.95 |      84.72 |     -86.13
   9    | RA Gain Error                  |      67.61 |      49.66 |     -71.77
   10   | RA Dynamic Gain                |      40.90 |      40.24 |     -40.82
   11   | AM Tone                        |      32.04 |      29.03 |    -125.69
   12   | Clipping                       |      97.52 |      82.29 |     -94.76
   13   | Drift                          |      37.40 |      43.77 |    -126.36
   14   | Reference Error                |      67.44 |      66.60 |     -66.84
   15   | Glitch                         |      70.18 |      38.22 |     -73.17
   ====================================================================================================
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a25_spectra.png]

---

exp_a31_fit_static_nonlin.py
----------------------------

**Description**: Fit static nonlinearity coefficients from sine wave.

.. code-block:: none

   [Sinewave] Fs=[800.00 MHz], Fin=[70.02 MHz], Bin/N=[717/8192], A=[0.500 Vpeak]
   [Nonideal] Noise RMS=[500.00 uVrms], Theoretical SNR=[56.99 dB], Theoretical NSD=[-143.01 dBFS/Hz]
   [Injected: k2= 0.0000, k3= 0.0000] [Extracted: k2= 0.0002, k3=-0.0001]
   [Injected: k2= 0.0100, k3= 0.0000] [Extracted: k2= 0.0099, k3=-0.0003]
   [Injected: k2= 0.0000, k3= 0.0100] [Extracted: k2= 0.0001, k3= 0.0097]
   [Injected: k2= 0.0100, k3= 0.0100] [Extracted: k2= 0.0099, k3= 0.0098]
   
   [Saved] -> D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a31_fit_static_nonlin.png

---

exp_a32_inl_from_sine_sweep_length.py
-------------------------------------

**Description**: Sweep data length for INL/DNL extraction from sine wave.

.. code-block:: none

   [INL/DNL Sweep] [Fs = 800 MHz, Fin = 80 MHz]
     [HD2 = -80 dB, HD3 = -66 dB, Noise = 50.0 uV]
   
     [N = 2^10 =  1024] [ENOB = 10.61] [INL: -110.85 to 118.80] [DNL: -1.00 to 196.37] LSB
     [N = 2^14 = 16384] [ENOB = 10.59] [INL: -28.50 to 40.99] [DNL: -1.00 to 17.49] LSB
     [N = 2^18 = 262144] [ENOB = 10.59] [INL: -22.56 to 30.00] [DNL: -1.00 to  2.83] LSB
     [N = 2^22 = 4194304] [ENOB = 10.59] [INL: -20.02 to 28.24] [DNL: -0.59 to  0.60] LSB
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a33_compute_inl_sweep_length.png]

---

exp_a41_analyze_phase_plane.py
------------------------------

**Description**: Analyze phase plane plots for various non-idealities.

.. code-block:: none

   [Config] Fs=800 MHz, Fin=97.0 MHz, Bin=7947, N=65536
   [Config] A=0.490 V, DC=0.500 V, Resolution=12 bits
   
   ====================================================================================================
   Case | Non-Ideality                   | Lag    | Outliers
   ----------------------------------------------------------------------------------------------------
   1    | Thermal Noise                  | 2      | 0
   2    | Quantization Noise             | 2      | 0
   3    | Jitter Noise                   | 2      | 0
   4    | AM Noise                       | 2      | 0
   5    | Static HD2 (-80 dBc)           | 2      | 0
   6    | Static HD3 (-70 dBc)           | 2      | 0
   7    | Memory Effect                  | 2      | 0
   8    | Incomplete Settling            | 2      | 0
   9    | RA Gain Error                  | 2      | 0
   10   | RA Dynamic Gain                | 2      | 0
   11   | AM Tone                        | 2      | 0
   12   | Clipping                       | 2      | 0
   13   | Drift                          | 2      | 0
   14   | Reference Error                | 2      | 0
   15   | Glitch                         | 2      | 164
   ====================================================================================================
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a41_analyze_phase_plane.png]

---

exp_a42_analyze_error_phase_plane.py
------------------------------------

**Description**: Analyze error phase plane (1000x more sensitive for detecting harmonics).

.. code-block:: none

   [Config] Fs=800 MHz, Fin=97.0 MHz, Bin=7947, N=65536
   [Config] A=0.490 V, DC=0.500 V, Resolution=12 bits
   
   
   ================================================================================
   Case | Non-Ideality                   | RMS Error    | Peak Error
   --------------------------------------------------------------------------------
   1    | Thermal Noise                  |      179.8 uV |      764.3 uV
   2    | Quantization Noise             |      280.2 uV |      515.5 uV
   3    | Jitter Noise                   |      423.0 uV |     2656.3 uV
   4    | AM Noise                       |      174.4 uV |      969.8 uV
   5    | Static HD2 (-80 dBc)           |       36.0 uV |       86.9 uV
   6    | Static HD3 (-70 dBc)           |      110.1 uV |      198.7 uV
   7    | Memory Effect                  |      171.2 uV |      446.2 uV
   8    | Incomplete Settling            |       20.3 uV |      671.8 uV
   9    | RA Gain Error                  |     1127.6 uV |     2309.9 uV
   10   | RA Dynamic Gain                |     3434.2 uV |     8073.5 uV
   11   | AM Tone                        |    12255.7 uV |    24510.5 uV
   12   | Clipping                       |       26.5 uV |      260.9 uV
   13   | Drift                          |     4465.6 uV |     9736.5 uV
   14   | Reference Error                |      162.0 uV |      347.2 uV
   15   | Glitch                         |     4633.2 uV |    99857.2 uV
   ================================================================================
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\04_debug_analog\output\exp_a42_analyze_error_phase_plane.png]
   
   ================================================================================
   Key Observations:
   --------------------------------------------------------------------------------
   * HD2/HD3 cases show characteristic parabola/S-curve shapes even at -80/-70 dBc
   * Clipping shows sharp breaks at signal extremes
   * Memory effect and RA gain errors create subtle curvature
   * Thermal/quantization noise appear as horizontal scatter (good linearity)
   * This method is 1000x more sensitive than regular phase planes for detecting harmonics
   ================================================================================

---
