"""
Unit Test: Verify alias function with known frequency folding

Purpose: Self-verify that alias function correctly computes frequency folding
         into Nyquist bands (NOT compared against MATLAB)
"""
import numpy as np
import pytest
from adctoolbox.common import calc_aliased_freq


def test_verify_alias_basic():
    """
    Verify alias function with known aliasing cases.

    Test strategy:
    1. Test fundamental aliasing rules (Fs=1000 Hz)
    2. Assert: Each input frequency aliases to expected value
    """
    Fs = 1000
    test_cases = [
        (100, 100),   # Zone 0: Direct
        (600, 400),   # Zone 1: Reflected
        (1000, 0),    # At Fs: Aliases to DC
        (1200, 200),  # Zone 2: Direct
        (1600, 400),  # Zone 3: Reflected
    ]

    print(f'\n[Verify Alias] [Fs={Fs} Hz]')

    for Fin, expected in test_cases:
        result = calc_aliased_freq(Fin, Fs)
        error = abs(result - expected)
        status = 'PASS' if error < 0.01 else 'FAIL'
        print(f'  [Fin={Fin:4d} Hz] -> [Alias={result:3.0f} Hz] [Expected={expected:3d} Hz] [{status}]')
        assert error < 0.01, f"Aliasing error: {Fin} Hz -> {result} Hz (expected {expected} Hz)"


def test_verify_alias_zones():
    """
    Verify alias function across multiple Nyquist zones.

    Test strategy:
    1. Sweep frequencies across 5 Nyquist zones
    2. Verify zone 0,2,4 (even): Direct mapping
    3. Verify zone 1,3 (odd): Reflected mapping
    """
    Fs = 1000
    Fnyq = Fs / 2

    print(f'\n[Verify Alias Zones] [Fs={Fs} Hz] [Fnyq={Fnyq} Hz]')

    # Test representative frequencies in each zone
    test_points = [
        # (Fin, zone, expected_alias, direction)
        (100, 0, 100, 'Direct'),      # Zone 0 (0-500 Hz)
        (400, 0, 400, 'Direct'),
        (600, 1, 400, 'Reflected'),   # Zone 1 (500-1000 Hz)
        (900, 1, 100, 'Reflected'),
        (1100, 2, 100, 'Direct'),     # Zone 2 (1000-1500 Hz)
        (1400, 2, 400, 'Direct'),
        (1600, 3, 400, 'Reflected'),  # Zone 3 (1500-2000 Hz)
        (1900, 3, 100, 'Reflected'),
    ]

    for Fin, zone, expected, direction in test_points:
        result = calc_aliased_freq(Fin, Fs)
        error = abs(result - expected)
        status = 'PASS' if error < 0.01 else 'FAIL'
        print(f'  [Zone {zone}] [Fin={Fin:4d} Hz] -> [{result:3.0f} Hz] ({direction:9s}) [{status}]')
        assert error < 0.01, f"Zone {zone} aliasing error: {Fin} Hz -> {result} Hz (expected {expected} Hz)"


def test_verify_alias_harmonics():
    """
    Verify harmonic aliasing pattern.

    Test strategy:
    1. Generate harmonics of 101 Hz with Fs=1024 Hz
    2. Verify each harmonic aliases correctly
    3. Check alternating direct/reflected pattern
    """
    Fin = 101
    Fs = 1024
    Fnyq = Fs / 2

    print(f'\n[Verify Harmonic Aliasing] [Fin={Fin} Hz] [Fs={Fs} Hz]')

    # Expected pattern for first 10 harmonics
    expected_aliases = [
        101,  # H1: 101 Hz (zone 0, direct)
        202,  # H2: 202 Hz (zone 0, direct)
        303,  # H3: 303 Hz (zone 0, direct)
        404,  # H4: 404 Hz (zone 0, direct)
        505,  # H5: 505 Hz (zone 0, direct)
        418,  # H6: 606 Hz (zone 1, reflected)
        317,  # H7: 707 Hz (zone 1, reflected)
        216,  # H8: 808 Hz (zone 1, reflected)
        115,  # H9: 909 Hz (zone 1, reflected)
        14,   # H10: 1010 Hz (zone 1, reflected)
    ]

    for h in range(1, 11):
        freq = Fin * h
        result = calc_aliased_freq(freq, Fs)
        expected = expected_aliases[h-1]
        error = abs(result - expected)
        zone = int(freq / Fnyq)
        status = 'PASS' if error < 1.0 else 'FAIL'
        print(f'  [H{h:2d}] [{freq:4d} Hz] [Zone {zone}] -> [{result:3.0f} Hz] [{status}]')
        assert error < 1.0, f"H{h} aliasing error: {freq} Hz -> {result} Hz (expected {expected} Hz)"


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('Running alias verification tests...\n')
    test_verify_alias_basic()
    test_verify_alias_zones()
    test_verify_alias_harmonics()
    print('\n** All alias verification tests passed! **')
