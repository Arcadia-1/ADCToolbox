"""Digital output toolset: run all 3 analysis tools on ADC digital codes"""
import numpy as np
from pathlib import Path
from adctoolbox.dout.generate_dout_dashboard import generate_dout_dashboard

# Load digital output data (e.g., from CSV file)
data_file = Path(__file__).parent.parent.parent.parent.parent / 'reference_dataset' / 'dout_SAR_12b_weight_2.csv'

if data_file.exists():
    bits = np.loadtxt(data_file, delimiter=',')
    print(f"[Loaded data] {data_file.name} ({bits.shape[0]} samples x {bits.shape[1]} bits)")
else:
    # Generate sample data if file not found
    print("[Generating sample data]")
    N = 2**13
    n_bits = 12
    
    # Generate ideal sine
    Fs = 800e6
    Fin = 80e6
    t = np.arange(N) / Fs
    A, DC = 0.49, 0.5
    signal = A * np.sin(2*np.pi*Fin*t) + DC
    
    # Quantize to bits (MSB to LSB)
    codes = np.round(signal * (2**n_bits - 1)).astype(int)
    codes = np.clip(codes, 0, 2**n_bits - 1)
    
    # Convert to binary bits
    bits = np.zeros((N, n_bits), dtype=int)
    for i in range(n_bits):
        bits[:, i] = (codes >> (n_bits - 1 - i)) & 1
    
    print(f"Generated {N} samples x {n_bits} bits")

# Output directory
output_dir = Path(__file__).parent / 'output' / 'toolset_dout'

print(f"\n=== Running DOUT Toolset (3 Tools) ===")
print(f"Output directory: {output_dir}\n")

# Run toolset
status = generate_dout_dashboard(
    bits,
    output_dir,
    visible=False,      # Set to True to display figures
    order=5,            # Polynomial order for calibration
    prefix='dout'       # Filename prefix
)

# Print summary
print(f"\n=== Summary ===")
print(f"Success: {status['success']}")
print(f"Tools completed: {sum(status['tools_completed'])}/3")
print(f"Panel: {status['panel_path']}")
if status['errors']:
    print(f"Errors: {status['errors']}")
