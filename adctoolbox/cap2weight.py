"""
Capacitor to Weight Converter - Ported from MATLAB cap2weight.m

    MSB <---||--------||---< LSB
          Cb   |    |   Cl
              ---  ---
          Cp  ---  ---  Cd
               |    |
              gnd   Vi
"""

import numpy as np


def cap2weight(cd, cb, cp):
    """
    Convert CDAC capacitor values to bit weights.

    Args:
        cd: DAC bit capacitors [LSB ... MSB]
        cb: Bridge capacitors [LSB ... MSB], 0 means no bridge
        cp: Parasitic capacitors [LSB ... MSB]

    Returns:
        weight: Gain from Vi to Vout [LSB ... MSB]
        co: Output capacitance
    """
    cd = np.asarray(cd, dtype=float)
    cb = np.asarray(cb, dtype=float)
    cp = np.asarray(cp, dtype=float)

    m = len(cd)
    weight = np.zeros(m)
    cl = 0.0

    for i in range(m):
        cs = cp[i] + cd[i] + cl
        weight = weight * cl / cs if cs > 0 else weight
        weight[i] = cd[i] / cs if cs > 0 else 0

        if cb[i] == 0:
            cl = cs
        else:
            cl = 1.0 / (1.0 / cb[i] + 1.0 / cs)

    return weight, cl
