"""
INL/DNL analysis and plotting from sine wave excitation.

This is a wrapper function that combines INL/DNL computation and plotting
for convenient use.
"""

from adctoolbox.aout.compute_inl_from_sine import compute_inl_from_sine
from adctoolbox.aout.plot_dnl_inl import plot_dnl_inl

def analyze_inl_from_sine(data, num_bits=None, full_scale=None, clip_percent=0.01,
                          create_plot: bool = True, show_title=True, col_title=None, ax=None):
    """
    INL/DNL analysis from sine wave excitation with optional plotting.

    This function computes INL and DNL from analog or digital signal data,
    and optionally plots the results.

    Parameters:
    -----------
    data : array_like
        Input signal - either analog voltage or digital codes.
        - Analog: Float values in range (normalized 0-1, or full-scale voltage)
        - Digital: Integer codes or float representation of codes
    num_bits : int, optional
        ADC number of bits. If None, infers from data range.
    full_scale : float, optional
        Full scale voltage for quantization. If provided with analog input,
        codes = round(data * 2^num_bits / full_scale).
        If None, assumes normalized input (0-1 range).
    clip_percent : float, default=0.01
        Percentage of codes to clip from edges (0.01 = 1% from each end)
    create_plot : bool, default=True
        Plot the INL/DNL curves (True) or just compute (False)
    show_title : bool, default=True
        Show auto-generated title with min/max ranges
    col_title : str, optional
        Column title to display above DNL plot (e.g., "N = 2^10")
    ax : matplotlib.axes.Axes, optional
        Single axis to split into 2 rows. If None and create_plot=True,
        uses current axis (plt.gca()).

    Returns:
    --------
    dict : Dictionary containing:
        - 'inl': INL values in LSB
        - 'dnl': DNL values in LSB
        - 'code': Code values (x-axis)
    """

    # 1. --- Core Calculation ---
    inl, dnl, code = compute_inl_from_sine(
        data=data,
        num_bits=num_bits,
        full_scale=full_scale,
        clip_percent=clip_percent
    )

    # 2. --- Optional Plotting ---
    if create_plot:
        plot_dnl_inl(
            code=code,
            dnl=dnl,
            inl=inl,
            num_bits=num_bits,
            show_title=show_title,
            col_title=col_title,
            ax=ax
        )

    # 3. --- Return Results ---
    return {
        'inl': inl,
        'dnl': dnl,
        'code': code
    }
