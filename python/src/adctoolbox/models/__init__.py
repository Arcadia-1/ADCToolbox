"""ADC behavioral models — forward operators (vin → codes).

Distinct from ``siggen`` (which generates analog test signals **before**
the ADC) and from ``dout``/``spectrum`` (which **analyze** code streams).
These are the discrete-time conversion models themselves.

Currently includes:
    - SAR (binary or sub-radix-2, optional cap mismatch, sampling noise,
      and comparator noise)

Convention (matches Arcadia-1/SpecMind reference):
    vin     is interpreted relative to quant_range; default quant_range is (0, 1)
    weights are normalized by sum(bit_weights) + 1 LSB
    codes   ∈ {0, 1}^B, MSB at index 0
    recon   = codes @ digital_weights, scaled by quant_range when requested
"""
from adctoolbox.models.sar import (
    sar_convert,
    sar_reconstruct,
    sar_ideal_weights,
    sar_apply_cap_mismatch,
    sar_apply_mismatch,
)

__all__ = [
    "sar_convert",
    "sar_reconstruct",
    "sar_ideal_weights",
    "sar_apply_cap_mismatch",
    "sar_apply_mismatch",
]
