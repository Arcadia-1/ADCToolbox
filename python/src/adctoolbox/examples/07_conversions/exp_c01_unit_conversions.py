"""
Comprehensive unit conversion examples for ADC testing: dB, power, voltage, LSB, frequency bins,
ENOB/SNR, and NSD. Tests bidirectional conversions to verify accuracy (forward and reverse).
"""
import numpy as np
from pathlib import Path
from adctoolbox import (db_to_mag, mag_to_db, db_to_power, power_to_db,
                         lsb_to_volts, volts_to_lsb, bin_to_freq, freq_to_bin,
                         snr_to_enob, enob_to_snr, snr_to_nsd, nsd_to_snr,
                         dbm_to_vrms, vrms_to_dbm, dbm_to_mw, mw_to_dbm,
                         sine_amplitude_to_power)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

print("[Unit Conversions for ADC Testing]\n")

# 1a. dB -> Magnitude -> dB
print("[1a. dB -> Magnitude -> dB]")
for db_val in [-80, -70, -60, -40, -20]:
    mag = db_to_mag(db_val)
    db_back = mag_to_db(mag)
    print(f"  [dB = {db_val:4d}] -> [mag = {mag:.6f}] -> [dB = {db_back:6.2f}]")

# 1b. Magnitude -> dB -> Magnitude
print("\n[1b. Magnitude -> dB -> Magnitude]")
for mag in [0.0001, 0.001, 0.01, 0.1, 1.0]:
    db_val = mag_to_db(mag)
    mag_back = db_to_mag(db_val)
    print(f"  [mag = {mag:.6f}] -> [dB = {db_val:6.2f}] -> [mag = {mag_back:.6f}]")

# 2a. dB -> Power -> dB
print("\n[2a. dB -> Power -> dB]")
for db_val in [0, 10, 20, 30, 40]:
    power = db_to_power(db_val)
    db_back = power_to_db(power)
    print(f"  [dB = {db_val:3d}] -> [power = {power:8.1f}x] -> [dB = {db_back:6.2f}]")

# 2b. Power -> dB -> Power
print("\n[2b. Power -> dB -> Power]")
for power in [1, 10, 100, 1000, 10000]:
    db_val = power_to_db(power)
    power_back = db_to_power(db_val)
    print(f"  [power = {power:5d}x] -> [dB = {db_val:6.2f}] -> [power = {power_back:8.1f}x]")

# 3a. dBm -> Vrms -> dBm (50 ohm)
print("\n[3a. dBm -> Vrms -> dBm (50 ohm)]")
for dbm in [-20, -10, 0, 10, 20]:
    vrms = dbm_to_vrms(dbm)
    dbm_back = vrms_to_dbm(vrms)
    print(f"  [dBm = {dbm:4d}] -> [Vrms = {vrms*1e3:7.2f} mV] -> [dBm = {dbm_back:6.2f}]")

# 3b. Vrms -> dBm -> Vrms (50 ohm)
print("\n[3b. Vrms -> dBm -> Vrms (50 ohm)]")
for vrms in [0.001, 0.01, 0.1, 0.316, 1.0]:
    dbm = vrms_to_dbm(vrms)
    vrms_back = dbm_to_vrms(dbm)
    print(f"  [Vrms = {vrms*1e3:6.1f} mV] -> [dBm = {dbm:6.2f}] -> [Vrms = {vrms_back*1e3:7.2f} mV]")

# 4a. dBm -> mW -> dBm
print("\n[4a. dBm -> mW -> dBm]")
for dbm in [-10, 0, 10, 20, 30]:
    mw = dbm_to_mw(dbm)
    dbm_back = mw_to_dbm(mw)
    print(f"  [dBm = {dbm:4d}] -> [mW = {mw:8.2f}] -> [dBm = {dbm_back:6.2f}]")

# 4b. mW -> dBm -> mW
print("\n[4b. mW -> dBm -> mW]")
for mw in [0.1, 1, 10, 100, 1000]:
    dbm = mw_to_dbm(mw)
    mw_back = dbm_to_mw(dbm)
    print(f"  [mW = {mw:7.1f}] -> [dBm = {dbm:6.2f}] -> [mW = {mw_back:8.2f}]")

# 5. Sine Amplitude -> Power
print("\n[5. Sine Wave Amplitude -> Power (50 ohm)]")
for amp in [0.1, 0.316, 0.5, 1.0, 2.0]:
    power_w = sine_amplitude_to_power(amp)
    print(f"  [Amplitude = {amp:5.3f} V] -> [Power = {power_w*1e3:7.3f} mW] = [{mw_to_dbm(power_w*1e3):6.2f} dBm]")

# 6a. Voltage -> LSB -> Voltage
print("\n[6a. Voltage -> LSB -> Voltage (12-bit ADC, VFS=1V)]")
vfs = 1.0
n_bits = 12
for volts in [100e-6, 250e-6, 500e-6, 1e-3, 2e-3]:
    lsb = volts_to_lsb(volts, vfs, n_bits)
    volts_back = lsb_to_volts(lsb, vfs, n_bits)
    print(f"  [V = {volts*1e6:6.1f} uV] -> [LSB = {lsb:6.2f}] -> [V = {volts_back*1e6:6.1f} uV]")

# 6b. LSB -> Voltage -> LSB
print("\n[6b. LSB -> Voltage -> LSB (12-bit ADC, VFS=1V)]")
for lsb in [0.5, 1, 2, 5, 10]:
    volts = lsb_to_volts(lsb, vfs, n_bits)
    lsb_back = volts_to_lsb(volts, vfs, n_bits)
    print(f"  [LSB = {lsb:6.2f}] -> [V = {volts*1e6:6.1f} uV] -> [LSB = {lsb_back:6.2f}]")

# 7a. Frequency -> Bin -> Frequency
print("\n[7a. Frequency -> Bin -> Frequency (Fs=100MHz, N=8192)]")
fs = 100e6
n_fft = 8192
for freq in [1e6, 5e6, 10e6, 20e6, 40e6]:
    bin_idx = freq_to_bin(freq, fs, n_fft)
    freq_back = bin_to_freq(bin_idx, fs, n_fft)
    print(f"  [Freq = {freq/1e6:5.1f} MHz] -> [Bin = {bin_idx:4d}] -> [Freq = {freq_back/1e6:5.2f} MHz]")

# 7b. Bin -> Frequency -> Bin (edge cases)
print("\n[7b. Bin -> Frequency -> Bin (Fs=100MHz, N=8192) - Edge cases]")
for bin_idx in [1, 2, 3, n_fft//2-1, n_fft//2]:
    freq = bin_to_freq(bin_idx, fs, n_fft)
    bin_back = freq_to_bin(freq, fs, n_fft)
    print(f"  [Bin = {bin_idx:4d}] -> [Freq = {freq/1e6:6.3f} MHz] -> [Bin = {bin_back:4d}]")

# 8a. SNDR -> ENOB -> SNDR
print("\n[8a. SNDR -> ENOB -> SNDR]")
for sndr in [50, 60, 70, 80, 90]:
    enob = snr_to_enob(sndr)
    sndr_back = enob_to_snr(enob)
    print(f"  [SNDR = {sndr:6.2f} dB] -> [ENOB = {enob:6.2f} bit] -> [SNDR = {sndr_back:6.2f} dB]")

# 8b. ENOB -> SNDR -> ENOB
print("\n[8b. ENOB -> SNDR -> ENOB]")
for enob in [8, 10, 12, 14, 16]:
    sndr = enob_to_snr(enob)
    enob_back = snr_to_enob(sndr)
    print(f"  [ENOB = {enob:6.2f} bit] -> [SNDR = {sndr:6.2f} dB] -> [ENOB = {enob_back:6.2f} bit]")

# 9a. SNDR -> NSD -> SNDR
print("\n[9a. SNDR -> NSD -> SNDR (Fs=800MHz, OSR=1)]")
fs = 800e6
osr = 1
for sndr in [60, 70, 80, 90, 100]:
    nsd = snr_to_nsd(sndr, fs, osr)
    sndr_back = nsd_to_snr(nsd, fs, osr)
    print(f"  [SNDR = {sndr:5.1f} dB] -> [NSD = {nsd:7.2f} dBFS/Hz] -> [SNDR = {sndr_back:5.2f} dB]")

# 9b. NSD -> SNDR -> NSD
print("\n[9b. NSD -> SNDR -> NSD (Fs=800MHz, OSR=1)]")
for nsd in [-170, -165, -160, -155, -150]:
    sndr = nsd_to_snr(nsd, fs, osr)
    nsd_back = snr_to_nsd(sndr, fs, osr)
    print(f"  [NSD = {nsd:4d} dBFS/Hz] -> [SNDR = {sndr:6.2f} dB] -> [NSD = {nsd_back:7.2f} dBFS/Hz]")

print(f"\n[Example complete]")
