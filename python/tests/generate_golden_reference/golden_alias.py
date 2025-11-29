"""golden_alias.py - Golden reference test for alias

This test generates alias report for golden reference.
"""

import sys
from pathlib import Path

# Get project root directory
project_root = Path(__file__).resolve().parents[3]

# Add unit tests to path
unit_tests_dir = project_root / 'python' / 'tests' / 'unit'
sys.path.insert(0, str(unit_tests_dir))

from adctoolbox.common import alias


def golden_alias():
    """Generate alias report for golden reference."""

    output_path = project_root / 'test_reference' / 'test_alias' / 'alias_report_python.txt'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"Frequency Aliasing Report (Fs=1000 Hz, Nyquist=500.0 Hz)")
    lines.append("=" * 60)
    lines.append(f"{'Fin':<10} {'Zone':<6} {'Aliased':<10} {'Direction'}")
    lines.append("-" * 60)

    # Test frequencies from 0 to 5*Fs
    Fs = 1000
    test_freqs = list(range(0, int(5*Fs) + 1, 100))

    for Fin in test_freqs:
        zone = int(Fin / (Fs / 2))
        f_alias = alias(Fin, Fs)
        direction = "Direct" if zone % 2 == 0 else "Reflected"
        lines.append(f"{Fin:<10} {zone:<6} {f_alias:<10.0f} {direction}")

    lines.append("-" * 60)

    # Harmonic example
    lines.append("\nHarmonic Example (Fin=101 Hz, Fs=1024 Hz):")
    lines.append(f"{'H':<4} {'Freq':<8} {'Zone':<6} {'Aliased'}")
    for h in range(1, 11):
        f = 101 * h
        zone = int(f / 512)
        lines.append(f"{h:<4} {f:<8} {zone:<6} {alias(f, 1024):.0f}")

    report = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"[golden_alias] saved to: {output_path}")


if __name__ == "__main__":
    golden_alias()
