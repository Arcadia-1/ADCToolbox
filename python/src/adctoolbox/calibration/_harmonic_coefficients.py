"""Decode the shared sine-solver layout, with one fixed coefficient overall."""

import numpy as np


def _unpack_harmonic_coefficients(solution, harmonic_start, num_harmonics, basis_choice):
    """Return full cosine/sine vectors across all records and harmonic orders."""
    quadrature_start = harmonic_start + num_harmonics - 1
    unity_side = np.concatenate(([1.0], solution[harmonic_start:quadrature_start]))
    other_side = solution[quadrature_start:quadrature_start + num_harmonics]
    if basis_choice == 0:
        return unity_side, other_side
    return other_side, unity_side
