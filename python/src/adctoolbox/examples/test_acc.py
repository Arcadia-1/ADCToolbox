"""Compare accuracy of binned vs raw decomposition methods"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

def decompose_binned(err, phase_vals, A, bin_count=100):
    """Old method: Bin errors first, then decompose"""
    phase_axis = np.arange(bin_count) / bin_count * 360

    # Binning
    esum_sq = np.zeros(bin_count)
    enum = np.zeros(bin_count)
    bin_indices = (phase_vals / 360 * bin_count).astype(int) % bin_count
    np.add.at(esum_sq, bin_indices, err**2)
    np.add.at(enum, bin_indices, 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        erms = np.sqrt(esum_sq / enum)

    # Decomposition on binned data
    rad_axis = phase_axis / 180 * np.pi
    asen = np.sin(rad_axis)**2
    psen = np.cos(rad_axis)**2

    valid_mask = ~np.isnan(erms) & (enum > 10)
    erms_squared = erms[valid_mask]**2
    asen_fit = asen[valid_mask]
    psen_fit = psen[valid_mask]

    A_matrix = np.column_stack([asen_fit, psen_fit])
    coeffs = np.linalg.lstsq(A_matrix, erms_squared, rcond=None)[0]

    calc_anoi = np.sqrt(max(coeffs[0], 0))
    calc_pnoi_rad = np.sqrt(max(coeffs[1], 0)) / A

    return calc_anoi, calc_pnoi_rad

def decompose_raw(err, phase_vals, A):
    """New method: Decompose directly on raw data"""
    rad_vals = phase_vals * np.pi / 180

    # Basis functions for every point
    asen_raw = np.sin(rad_vals)**2
    psen_raw = np.cos(rad_vals)**2

    # Least squares on raw data
    A_matrix = np.column_stack([asen_raw, psen_raw])
    coeffs = np.linalg.lstsq(A_matrix, err**2, rcond=None)[0]

    calc_anoi = np.sqrt(max(coeffs[0], 0))
    calc_pnoi_rad = np.sqrt(max(coeffs[1], 0)) / A

    return calc_anoi, calc_pnoi_rad

def generate_signal(N, Fs, Fin, A, DC, amp_noise, phase_noise, thermal_noise, seed=None):
    """Generate test signal with known noise parameters"""
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(N) / Fs
    phase_arg = 2*np.pi*Fin*t

    # Generate noise
    n_amp = np.random.randn(N) * amp_noise
    n_phase = np.random.randn(N) * phase_noise
    n_thermal = np.random.randn(N) * thermal_noise

    # Create signal
    signal_clean = A * np.sin(phase_arg) + DC
    signal_noisy = (A + n_amp) * np.sin(phase_arg + n_phase) + DC + n_thermal
    err = signal_noisy - signal_clean

    phase_vals = np.mod(phase_arg * 180 / np.pi, 360)

    return err, phase_vals

# Test parameters
Fs = 800e6
A, DC = 0.5, 0.0

# Test scenarios
scenarios = [
    # (N, amp_noise, phase_noise, thermal_noise, description)
    (2**13, 100e-6, 1000e-6, 50e-6, "Small N, PM>>AM"),
    (2**16, 100e-6, 1000e-6, 50e-6, "Medium N, PM>>AM"),
    (2**19, 100e-6, 1000e-6, 50e-6, "Large N, PM>>AM"),
    (2**16, 1000e-6, 100e-6, 50e-6, "Medium N, AM>>PM"),
    (2**16, 500e-6, 500e-6, 50e-6, "Medium N, AM=PM"),
    (2**16, 50e-6, 0, 50e-6, "Medium N, Pure AM"),
    (2**16, 0, 1000e-6, 50e-6, "Medium N, Pure PM"),
]

results = []

print("[Accuracy Comparison: Binned vs Raw Decomposition]")
print("="*100)
print(f"{'Scenario':<20} {'N':>8} {'Method':<8} {'AM True':>10} {'AM Calc':>10} {'AM Err%':>10} {'PM True':>10} {'PM Calc':>10} {'PM Err%':>10} {'Time':>8}")
print("="*100)

for N, amp_noise, phase_noise, thermal_noise, desc in scenarios:
    Fin = 123/N * Fs

    # Generate signal
    err, phase_vals = generate_signal(N, Fs, Fin, A, DC, amp_noise, phase_noise, thermal_noise, seed=42)

    # Method 1: Binned
    t0 = time.time()
    anoi_binned, pnoi_binned = decompose_binned(err, phase_vals, A, bin_count=100)
    t_binned = time.time() - t0

    # Method 2: Raw
    t0 = time.time()
    anoi_raw, pnoi_raw = decompose_raw(err, phase_vals, A)
    t_raw = time.time() - t0

    # Calculate errors
    am_err_binned = (anoi_binned - amp_noise) / amp_noise * 100 if amp_noise > 0 else 0
    pm_err_binned = (pnoi_binned - phase_noise) / phase_noise * 100 if phase_noise > 0 else 0
    am_err_raw = (anoi_raw - amp_noise) / amp_noise * 100 if amp_noise > 0 else 0
    pm_err_raw = (pnoi_raw - phase_noise) / phase_noise * 100 if phase_noise > 0 else 0

    # Store results
    results.append({
        'desc': desc,
        'N': N,
        'amp_true': amp_noise,
        'phase_true': phase_noise,
        'anoi_binned': anoi_binned,
        'pnoi_binned': pnoi_binned,
        'anoi_raw': anoi_raw,
        'pnoi_raw': pnoi_raw,
        'am_err_binned': am_err_binned,
        'pm_err_binned': pm_err_binned,
        'am_err_raw': am_err_raw,
        'pm_err_raw': pm_err_raw,
        't_binned': t_binned,
        't_raw': t_raw
    })

    # Print results
    print(f"{desc:<20} {N:>8} {'Binned':<8} {amp_noise*1e6:>10.1f} {anoi_binned*1e6:>10.1f} {am_err_binned:>+9.1f}% {phase_noise*1e6:>10.1f} {pnoi_binned*1e6:>10.1f} {pm_err_binned:>+9.1f}% {t_binned*1000:>7.1f}ms")
    print(f"{'':<20} {N:>8} {'Raw':<8} {amp_noise*1e6:>10.1f} {anoi_raw*1e6:>10.1f} {am_err_raw:>+9.1f}% {phase_noise*1e6:>10.1f} {pnoi_raw*1e6:>10.1f} {pm_err_raw:>+9.1f}% {t_raw*1000:>7.1f}ms")
    print("-"*100)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Amplitude noise error vs N
scenarios_pm_gt_am = [r for r in results if 'PM>>AM' in r['desc']]
N_vals = [r['N'] for r in scenarios_pm_gt_am]
am_err_binned = [r['am_err_binned'] for r in scenarios_pm_gt_am]
am_err_raw = [r['am_err_raw'] for r in scenarios_pm_gt_am]

axes[0, 0].plot(N_vals, am_err_binned, 'b-o', linewidth=2, markersize=8, label='Binned')
axes[0, 0].plot(N_vals, am_err_raw, 'r-s', linewidth=2, markersize=8, label='Raw')
axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].set_xlabel('Sample Count (N)')
axes[0, 0].set_ylabel('Amplitude Noise Error (%)')
axes[0, 0].set_title('AM Error vs Sample Count (PM>>AM scenario)')
axes[0, 0].set_xscale('log')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Phase noise error vs N
pm_err_binned = [r['pm_err_binned'] for r in scenarios_pm_gt_am]
pm_err_raw = [r['pm_err_raw'] for r in scenarios_pm_gt_am]

axes[0, 1].plot(N_vals, pm_err_binned, 'b-o', linewidth=2, markersize=8, label='Binned')
axes[0, 1].plot(N_vals, pm_err_raw, 'r-s', linewidth=2, markersize=8, label='Raw')
axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].set_xlabel('Sample Count (N)')
axes[0, 1].set_ylabel('Phase Noise Error (%)')
axes[0, 1].set_title('PM Error vs Sample Count (PM>>AM scenario)')
axes[0, 1].set_xscale('log')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Error comparison across scenarios (N=2^16)
scenarios_medium_n = [r for r in results if r['N'] == 2**16]
labels = [r['desc'].replace('Medium N, ', '') for r in scenarios_medium_n]
am_errs_binned = [r['am_err_binned'] for r in scenarios_medium_n]
am_errs_raw = [r['am_err_raw'] for r in scenarios_medium_n]

x = np.arange(len(labels))
width = 0.35

axes[1, 0].bar(x - width/2, am_errs_binned, width, label='Binned', color='b', alpha=0.7)
axes[1, 0].bar(x + width/2, am_errs_raw, width, label='Raw', color='r', alpha=0.7)
axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].set_xlabel('Scenario')
axes[1, 0].set_ylabel('Amplitude Noise Error (%)')
axes[1, 0].set_title('AM Error Across Scenarios (N=2^16)')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Phase noise error comparison
pm_errs_binned = [r['pm_err_binned'] for r in scenarios_medium_n]
pm_errs_raw = [r['pm_err_raw'] for r in scenarios_medium_n]

axes[1, 1].bar(x - width/2, pm_errs_binned, width, label='Binned', color='b', alpha=0.7)
axes[1, 1].bar(x + width/2, pm_errs_raw, width, label='Raw', color='r', alpha=0.7)
axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_xlabel('Scenario')
axes[1, 1].set_ylabel('Phase Noise Error (%)')
axes[1, 1].set_title('PM Error Across Scenarios (N=2^16)')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_path = output_dir / 'test_acc_comparison.png'
plt.savefig(fig_path, dpi=150)
print(f"\n[Save fig] -> [{fig_path}]")
plt.close()

# Summary statistics
print("\n[Summary Statistics]")
print(f"Average |AM Error| - Binned: {np.mean([abs(r['am_err_binned']) for r in results]):.2f}%")
print(f"Average |AM Error| - Raw:    {np.mean([abs(r['am_err_raw']) for r in results]):.2f}%")
print(f"Average |PM Error| - Binned: {np.mean([abs(r['pm_err_binned']) for r in results if r['phase_true'] > 0]):.2f}%")
print(f"Average |PM Error| - Raw:    {np.mean([abs(r['pm_err_raw']) for r in results if r['phase_true'] > 0]):.2f}%")
print(f"Average Time - Binned: {np.mean([r['t_binned'] for r in results])*1000:.2f} ms")
print(f"Average Time - Raw:    {np.mean([r['t_raw'] for r in results])*1000:.2f} ms")
