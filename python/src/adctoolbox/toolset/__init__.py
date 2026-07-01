"""
Toolset subpackage: Dashboard generation utilities.

This subpackage provides high-level dashboard functions that combine
multiple analysis tools into comprehensive visualizations.
"""

from adctoolbox.toolset.generate_aout_dashboard import (
    generate_aout_dashboard as generate_aout_dashboard_2x4,
)
from adctoolbox.toolset.generate_aout_dashboard_3x4 import (
    generate_aout_dashboard as generate_aout_dashboard_3x4,
)
from adctoolbox.toolset.generate_dout_dashboard import generate_dout_dashboard

generate_aout_dashboard = generate_aout_dashboard_3x4

__all__ = [
    'generate_aout_dashboard',
    'generate_aout_dashboard_2x4',
    'generate_aout_dashboard_3x4',
    'generate_dout_dashboard',
]
