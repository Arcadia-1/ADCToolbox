"""
Toolset subpackage: Dashboard generation utilities.

This subpackage provides high-level dashboard functions that combine
multiple analysis tools into comprehensive visualizations.
"""

from adctoolbox.toolset.generate_aout_dashboard import generate_aout_dashboard
from adctoolbox.toolset.generate_dout_dashboard import generate_dout_dashboard

__all__ = [
    'generate_aout_dashboard',
    'generate_dout_dashboard',
]
