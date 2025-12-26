Expected Output: 05_debug_digital
=======================================

This document shows the expected console output and example figures from all examples in `python/src/adctoolbox/examples/05_debug_digital/`.

Summary
-------

All examples in `05_debug_digital` demonstrate digital output analysis capabilities:

**Total Examples**: 7

**Categories**:
- **Weight Calibration**: exp_d01-d03 (lite, full, redundancy comparison)
- **Digital Debugging**: exp_d11-d14 (bit activity, ENOB sweep, weight scaling, overflow check)

**Key Features**:
- Foreground calibration using sine waves
- Redundancy analysis for pipeline ADCs
- Bit-level activity monitoring
- Weight and radix calculations
- Overflow detection


---

exp_d01_cal_weight_sine_lite.py
-------------------------------

**Description**: Lightweight weight calibration using sine wave.

.. code-block:: none

   [Config] N=8192, Fs=1000 MHz, Fin=79.96 MHz, Bin=655
   [Signal] Bit_width=12, MSB_mismatch=-1%, Quantized range=[4, 4091]
   
   [Spectrum Before] ENOB= 8.05 b, SNDR= 50.21 dB, SFDR= 53.48 dB, SNR= 54.70 dB, NSD=-141.68 dBFS/Hz
   [Spectrum Before] Noise Floor= -54.69 dBFS, Signal Power=  0.01 dBFS
   
   [Spectrum After]  ENOB=11.99 b, SNDR= 73.95 dB, SFDR= 95.26 dB, SNR= 73.95 dB, NSD=-160.94 dBFS/Hz
   [Spectrum After]  Noise Floor= -73.95 dBFS, Signal Power= -0.00 dBFS
   
   [Weight Recovery]
     True weights     : [2048.0, 1024.0, 512.0, 256.0, 128.0,  64.0,  32.0,  16.0,   8.0,   4.0,   2.0,   1.0]
     Recovered weights: [2042.3, 1031.5, 515.8, 257.8, 128.9,  64.4,  32.2,  16.1,   8.1,   4.0,   2.0,   1.0]
   
   [Performance]
     Calibration Runtime: 10.51 ms
     SNDR: 50.21 dB -> 73.95 dB (calc: 12.94 dB)
     ENOB: 8.05 bit -> 11.99 bit (calc: 1.86 bit)
     Improvement: +23.74 dB, +3.94 bit
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\05_debug_digital\output\exp_d01_cal_weight_sine_lite.png]

---

exp_d02_cal_weight_sine.py
--------------------------

**Description**: Full weight calibration with detailed analysis.

.. code-block:: none

   [Nominal Resolution] 12 bits
   [Weight Calibration] [ENoB =  8.05 bit] -> [ENoB = 12.02 bit]
     [Nominal]: [0.50000, 0.25000, 0.12500, 0.06250, 0.03125, 0.01562, 0.00781, 0.00391, 0.00195, 0.00098, 0.00049, 0.00024]
     [Real   ]: [0.49749, 0.25126, 0.12563, 0.06281, 0.03141, 0.01570, 0.00785, 0.00393, 0.00196, 0.00098, 0.00049, 0.00025] <-- Truth
     [Cal    ]: [0.49848, 0.25176, 0.12588, 0.06294, 0.03147, 0.01574, 0.00787, 0.00394, 0.00198, 0.00099, 0.00049, 0.00025] <-- Result
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\05_debug_digital\output\exp_d02_cal_weight_sine.png]
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\05_debug_digital\output\exp_d02_weight_error_comparison.png]

---

exp_d03_redundancy_comparison.py
--------------------------------

**Description**: Compare calibration with and without redundancy.

.. code-block:: none

   [No Redundancy (MSB -2%)            ] [Nominal = 12.00 bit] [ENoB =  7.03 bit] -> [ENoB = 11.93 bit]
   [No Redundancy (MSB +2%)            ] [Nominal = 12.00 bit] [ENoB =  7.04 bit] -> [ENoB = 10.20 bit]
   [With Redundancy (MSB -2%)          ] [Nominal = 12.17 bit] [ENoB =  7.20 bit] -> [ENoB = 12.13 bit]
   [With Redundancy (MSB +2%)          ] [Nominal = 12.17 bit] [ENoB =  7.19 bit] -> [ENoB = 12.18 bit]
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\05_debug_digital\output\exp_d03_redundancy_comparison.png]
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\05_debug_digital\output\exp_d03_weight_error_comparison.png]

---

exp_d11_bit_activity.py
-----------------------

**Description**: Check bit activity for digital debugging.

.. code-block:: none

   [ideal                   ] [Bits = 12] [ENoB = 12.31] [Activity = 50.0% - 50.0%]
   [+1% DC Offset           ] [Bits = 12] [ENoB =  9.82] [Activity = 50.3% - 53.3%]
   [-1% DC Offset           ] [Bits = 12] [ENoB =  9.82] [Activity = 46.7% - 49.7%]
   [Poor contact in Bit-11  ] [Bits = 12] [ENoB = 11.21] [Activity = 45.1% - 50.0%]
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\05_debug_digital\output\exp_d11_bit_activity.png]

---

exp_d12_sweep_bit_enob.py
-------------------------

**Description**: Sweep bit count to find optimal ENOB.

.. code-block:: none

   [Binary ADC with Thermal Noise                ] Runtime=  21.62ms, Nominal=12.00 bit, Max_ENOB=11.20 bit @ 12 bits, Final_ENOB=11.20 bit
   [Binary ADC with Thermal Noise + LSB random   ] Runtime=  21.78ms, Nominal=12.00 bit, Max_ENOB=10.69 bit @ 11 bits, Final_ENOB=10.69 bit
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\05_debug_digital\output\exp_d12_sweep_bit_enob.png]

---

exp_d13_weight_scaling.py
-------------------------

**Description**: Analyze weight scaling and radix calculations.

.. code-block:: none

   [Strict Binary Weights ] [Resolution = 12.00 bit] [Average Radix = 2.0000]
   [Sub-Radix-2 Weights   ] [Resolution = 12.34 bit] [Average Radix = 1.8196]
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\05_debug_digital\output\exp_d13_weight_scaling.png]

---

exp_d14_overflow_check.py
-------------------------

**Description**: Check for overflow in digital output codes.

.. code-block:: none

   [Binary ADC, Normal Range                ] Runtime= 33.62ms, MSB_range=[0.010, 0.990], Span=0.980
   [Binary ADC, Large Signal                ] Runtime= 57.06ms, MSB_range=[0.000, 1.000], Span=1.000
   [Sub-Radix with Redundancy               ] Runtime= 36.52ms, MSB_range=[0.010, 0.990], Span=0.980
   [Sub-Radix, Insufficient Redundancy      ] Runtime= 34.90ms, MSB_range=[0.010, 0.990], Span=0.980
   
   [Save fig] -> [D:\ADCToolbox\python\src\adctoolbox\examples\05_debug_digital\output\exp_d14_overflow_check.png]

---
