"""
Capacitor to Weight Converter.

Calculates the effective bit weights of a Capacitor DAC (CDAC)
based on component values, including parasitic and bridge capacitors.

Topology: Split-Capacitor Array (LSB -> Bridge -> MSB)
Ref: Ported from MATLAB cap2weight.m
"""

import numpy as np

def convert_cap_to_weight(
    caps_bit: np.ndarray,
    caps_bridge: np.ndarray,
    caps_parasitic: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Calculate bit weights for a CDAC with bridge capacitors.

    The algorithm iterates from LSB to MSB, calculating the equivalent
    load capacitance and scaling previous weights accordingly.

    Parameters
    ----------
    caps_bit : np.ndarray
        DAC bit capacitors [LSB ... MSB] (Cd)
    caps_bridge : np.ndarray
        Bridge capacitors [LSB ... MSB] (Cb). 0 indicates no bridge
    caps_parasitic : np.ndarray
        Parasitic capacitors to ground [LSB ... MSB] (Cp)

    Returns
    -------
    tuple[np.ndarray, float]
        - weights: Normalized weights [LSB ... MSB]
        - c_total: Total equivalent input capacitance
    """
    # 1. Input Validation & Standardization
    cd = np.asarray(caps_bit, dtype=float)
    cb = np.asarray(caps_bridge, dtype=float)
    cp = np.asarray(caps_parasitic, dtype=float)

    if not (len(cd) == len(cb) == len(cp)):
        raise ValueError(
            f"Input array lengths mismatch: Cd={len(cd)}, Cb={len(cb)}, Cp={len(cp)}"
        )

    n_bits = len(cd)
    weights = np.zeros(n_bits)
    
    # c_load represents the equivalent capacitance looking back towards LSB
    c_load = 0.0

    # 2. Iterative Calculation (LSB -> MSB)
    for i in range(n_bits):
        # Total capacitance at node i (Bit Cap + Parasitic + Load form previous stage)
        c_node_total = cd[i] + cp[i] + c_load
        
        if c_node_total == 0:
            # Avoid division by zero for disconnected nodes
            continue

        # attenuation_factor: The ratio of voltage division by the bridge
        # This scales ALL previous lower bits relative to the current bit
        attenuation_factor = c_load / c_node_total
        
        # Apply attenuation to all lower bits (0 to i-1)
        # This reflects that lower bits are "further away" through bridges
        weights[:i] *= attenuation_factor
        
        # Calculate weight of current bit i
        # Current bit contributes directly to the node voltage
        weights[i] = cd[i] / c_node_total

        # 3. Update Load Capacitance for next stage (i+1)
        # If there is a bridge cap (Cb > 0), calculate series combination
        if cb[i] > 0:
            # Series combination of Bridge and Node Total
            # C_next = Cb || C_node_total = (Cb * C_node) / (Cb + C_node)
            c_load = (cb[i] * c_node_total) / (cb[i] + c_node_total)
        else:
            # No bridge means direct connection (or simply summing up next)
            # Typically Cb=0 implies a segmented array block boundary or just simple summing
            c_load = c_node_total

    return weights, c_load