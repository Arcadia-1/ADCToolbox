"""
Signal Generation Module for ADC Testing

Provides signal generators and non-ideality appliers for simulating
various ADC imperfections and effects.
"""

from .nonidealities import ADC_Signal_Generator

__all__ = ["ADC_Signal_Generator"]
