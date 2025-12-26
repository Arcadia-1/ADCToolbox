"""
Target Vpp Calculator.

Calculates the required input amplitude (Vpp) to achieve a specific target
signal level (dBFS). Useful for automated instrument control loops.
"""

from adctoolbox.common.unit_conversions import db_to_mag

def vpp_for_target_dbfs(vpp_current: float, signal_db_measured: float, signal_db_target: float = -0.5) -> float:
    """
    Calculate the required Vpp to achieve a target dBFS level.

    Uses linear scaling based on the dB difference between the measured signal
    and the target signal.

    Args:
        vpp_current: The current input voltage peak-to-peak (V).
        signal_db_measured: The currently measured signal power (dBFS).
        signal_db_target: The desired signal power (dBFS), default -0.5.

    Returns:
        vpp_new: The calculated Vpp required to hit the target (V).
    """
    # Calculate how far off we are (in dB)
    delta_db = signal_db_target - signal_db_measured

    # Convert dB difference to a linear voltage gain ratio
    gain_ratio = db_to_mag(delta_db)

    # Apply gain to current voltage
    return vpp_current * gain_ratio
