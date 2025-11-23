"""
Test alias.py - Frequency Aliasing Analysis
Generates txt file showing how Fin values alias into Nyquist bands.
"""

import sys
import os

from ADC_Toolbox_Python.alias import alias


def generate_alias_report(Fs=1000, output_path=None):
    """Generate aliasing report for different input frequencies."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'output', 'alias_report.txt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append(f"Frequency Aliasing Report (Fs={Fs} Hz, Nyquist={Fs/2} Hz)")
    lines.append("=" * 60)
    lines.append(f"{'Fin':<10} {'Zone':<6} {'Aliased':<10} {'Direction'}")
    lines.append("-" * 60)

    # Test frequencies from 0 to 5*Fs
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

    print(f"[Alias report] saved to: {output_path}")
    return report


if __name__ == "__main__":
    # Quick verification
    Fs = 1000
    tests = [(100, 100), (600, 400), (1000, 0), (1200, 200), (1600, 400)]

    print(f"Testing alias.py (Fs={Fs} Hz)")
    for Fin, expected in tests:
        result = alias(Fin, Fs)
        status = "PASS" if abs(result - expected) < 0.01 else "FAIL"
        print(f"  {status} {Fin} Hz -> {result:.0f} Hz")

    print("")
    generate_alias_report(Fs=1000)
