"""Analog output toolset: run all 9 analysis tools on calibrated ADC data"""
import numpy as np
from pathlib import Path
from adctoolbox.aout.generate_aout_dashboard import generate_aout_dashboard

# Load analog output data (e.g., from CSV file)
data_file = Path(__file__).parent.parent.parent.parent.parent / 'reference_dataset' / 'sinewave_noise_200uV.csv'

if data_file.exists():
    aout_data = np.loadtxt(data_file, delimiter=',')
    print(f"[Loaded data] {data_file.name} ({len(aout_data)} samples)")
else:
    # Generate sample data if file not found
    print("[Generating sample data]")
    N = 2**13
    Fs = 800e6
    Fin = 80e6
    t = np.arange(N) / Fs
    A, DC = 0.49, 0.5
    noise_rms = 200e-6
    aout_data = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms
    print(f"Generated {N} samples")

# Output directory
output_dir = Path(__file__).parent / 'output' / 'toolset_aout'

print(f"\n=== Running AOUT Toolset (9 Tools) ===")
print(f"Output directory: {output_dir}\n")

# Run toolset
status = generate_aout_dashboard(
    aout_data, 
    output_dir, 
    visible=False,      # Set to True to display figures
    resolution=11,      # ADC resolution in bits
    prefix='aout'       # Filename prefix
)

# Print summary
print(f"\n=== Summary ===")
print(f"Success: {status['success']}")
print(f"Tools completed: {sum(status['tools_completed'])}/9")
print(f"Panel: {status['panel_path']}")
if status['errors']:
    print(f"Errors: {status['errors']}")
