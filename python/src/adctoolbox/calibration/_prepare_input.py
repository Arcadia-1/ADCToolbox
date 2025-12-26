"""
Prepare and preprocess calibration input data.
This module provides functions to format (reshape and stack) input data, set up nominal weights,
and scale bit columns for numerical conditioning.
"""

import numpy as np

def _prepare_input(
    bits_input: np.ndarray | list[np.ndarray],
    weights_input: np.ndarray | None = None,
    verbose: int = 0
) -> dict:
    """
    Orchestrate the full input preparation flow.
    """
    # 1. Normalize data: Matrix & segments
    bits_stacked, segment_lengths, bit_width, bits_segments = _reshape_and_stack_input(bits_input, verbose)
    num_segments = len(segment_lengths)

    # 2. Setup nominal weights
    nominal_weights = _setup_nominal_weights(bit_width, weights_input)

    # Return as a plain dictionary ordered by logical priority
    return {
        "bits_stacked": bits_stacked,    # (Ntot x M) stacked matrix
        "bits_segments": bits_segments,  # List for segment-wise access
        "bit_width": bit_width,          # M bits
        "num_segments": num_segments,    # N      
        "segment_lengths": segment_lengths, # Samples per set
        "nominal_weights": nominal_weights, # Nominal weights array
    }

def _reshape_and_stack_input(
    bits_input: np.ndarray | list[np.ndarray],
    verbose: int = 0
) -> tuple[np.ndarray, np.ndarray, int, list[np.ndarray]]:
    """
    Normalize input data: validate, transpose, and concatenate multi-dataset.

    This function handles both single ndarray and list of ndarrays, automatically
    transposing to ensure samples are rows, validating bitwidth consistency,
    and concatenating multiple datasets vertically.

    Parameters
    ----------
    bits_input : ndarray or list of ndarray
        Single bits matrix or list of bits matrices

    Returns
    -------
    bits_stacked : ndarray
        Concatenated bits matrix (Ntot x bit_width)
    segment_lengths : ndarray
        Number of samples per dataset
    bit_width : int
        Common bitwidth across all datasets
    bits_segments : list of ndarray
        Individual processed datasets (for later reconstruction)

    Raises
    ------
    ValueError
        If datasets are empty or have inconsistent bitwidths
    """
    # Unify input format to list of ndarrays (single dataset case -> list of one)
    if not isinstance(bits_input, list):
        bits_list = [bits_input]
    else:
        bits_list = bits_input

    if len(bits_list) == 0:
        raise ValueError('Empty input: bits_input cannot be an empty list.')

    bits_segments = []
    segment_lengths_list = []
    bit_width = None

    # Process each dataset: validate and transpose if needed
    for k, segment in enumerate(bits_list):
        if segment.size == 0:
            raise ValueError(f'Dataset {k} is empty.')

        # Check shape
        # N = number of samples (data points), M = bitwidth (bits/segments)
        # If rows < cols, automatically transpose (will print info if verbose)
        N_ori, M_ori = segment.shape
        if N_ori < M_ori:
            segment = segment.T
            if verbose >= 1:
                print(f"[INFO] Dataset [{k}]: Detected input shape ({N_ori}, {M_ori}). "
                      f"Automatically transposed to ({M_ori}, {N_ori}) to match (samples, bits).")

        N, M = segment.shape  # Update after potential transpose
        if bit_width is None:
            bit_width = M  # Set bitwidth from first dataset
        elif M != bit_width: 
            raise ValueError( # Fail fast on inconsistent bitwidths
                f'Inconsistent bitwidths: Dataset [{k}] has [{M}] bits, but previous datasets had [{bit_width}] bits.'
            )

        bits_segments.append(segment)
        segment_lengths_list.append(N) # Number of samples (after transpose)
        if verbose >= 2:
            print(f"[DEBUG] Add dataset [{k}] with shape ({N}, {M}). Total samples so far: {sum(segment_lengths_list)}.")

    bits_stacked = np.vstack(bits_segments)
    segment_lengths = np.array(segment_lengths_list, dtype=int)

    return bits_stacked, segment_lengths, bit_width, bits_segments

def _setup_nominal_weights(
    bit_width: int, 
    nominal_weights: np.ndarray | list | None = None
) -> np.ndarray:
    """
    Ensure nominal weights are set and have the correct shape.

    Parameters
    ----------
    bit_width : int
        Number of bits (M).
    nominal_weights : np.ndarray | list | None
        User-provided initial weights. Defaults to radix-2 if None.

    Returns
    -------
    weights : np.ndarray
        Validated weight array of length bit_width.
    """
    # Default to standard radix-2 weights: [2^(M-1), 2^(M-2), ..., 1]
    if nominal_weights is None:
        return 2.0 ** np.arange(bit_width - 1, -1, -1)
    
    # Standardize input to float array
    weights = np.asanyarray(nominal_weights, dtype=float)
    
    # Ensure physical consistency with the ADC bit-width
    if weights.size != bit_width:
        raise ValueError(
            f"Nominal weights length [{weights.size}] must match bit_width [{bit_width}]."
        )
        
    return weights
