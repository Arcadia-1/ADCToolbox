"""
test_alias.py - Unit test for alias function

Generates txt file showing how Fin values alias into Fnyquist bands.

Output structure:
    test_output/test_alias/
        alias_report.txt - Frequency aliasing report
        alias_results.csv - Numerical results for verification
"""

import numpy as np
from adctoolbox.common import alias
from tests._utils import save_variable

def test_alias(project_root):
    """Generate aliasing report for different input frequencies."""
    test_output_dir = project_root / "test_output" / "test_alias"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Test output directory: [{test_output_dir}]")

    Fs = 1000
    Fnyq = Fs / 2

    # Quick verification tests
    tests = [(100, 100), (600, 400), (1000, 0), (1200, 200), (1600, 400)]
    print(f"Testing alias function (Fs={Fs} Hz):")
    for Fin, expected in tests:
        result = alias(Fin, Fs)
        status = "✓" if abs(result - expected) < 0.01 else "✗"
        print(f"  {status} alias({Fin}, {Fs}) = {result:.0f} Hz (expected {expected})")

    # Generate comprehensive aliasing table
    test_freqs = np.arange(0, 5*Fs + 1, 100)
    zones = (test_freqs / Fnyq).astype(int)
    aliased = np.array([alias(f, Fs) for f in test_freqs])
    directions = np.where(zones % 2 == 0, "Direct", "Reflected")

    # Build report
    lines = [
        f"Frequency Aliasing Report (Fs={Fs} Hz, Fnyquist={Fnyq} Hz)",
        "=" * 60,
        f"{'Fin':<10} {'Zone':<6} {'Aliased':<10} {'Direction'}",
        "-" * 60
    ]

    for f, z, a, d in zip(test_freqs, zones, aliased, directions):
        lines.append(f"{f:<10.0f} {z:<6} {a:<10.0f} {d}")

    lines.append("-" * 60)

    # Harmonic example
    lines.extend([
        "\nHarmonic Example (Fin=101 Hz, Fs=1024 Hz):",
        f"{'H':<4} {'Freq':<8} {'Zone':<6} {'Aliased'}"
    ])

    for h in range(1, 11):
        f = 101 * h
        zone = int(f / 512)
        lines.append(f"{h:<4} {f:<8} {zone:<6} {alias(f, 1024):.0f}")

    # Save report
    report = "\n".join(lines)
    (test_output_dir / 'alias_report.txt').write_text(report)
    print(f"[Alias report] saved to: {test_output_dir / 'alias_report.txt'}")

    # Save numerical results
    alias_array = np.column_stack([test_freqs, zones, aliased])
    save_variable(test_output_dir, alias_array, 'alias_results')

