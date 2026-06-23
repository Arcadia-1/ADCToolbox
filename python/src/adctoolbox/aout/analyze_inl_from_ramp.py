"""
Ramp-based INL/DNL analysis and plotting.

This wrapper combines ramp-histogram INL/DNL computation with the existing
DNL/INL plot style used by the sine-histogram analyzer.
"""

from __future__ import annotations

from adctoolbox.aout.compute_inl_from_ramp import compute_inl_from_ramp
from adctoolbox.aout.plot_dnl_inl import plot_dnl_inl


def analyze_inl_from_ramp(
    codes,
    num_bits: int | None = None,
    code_min: int = 0,
    code_max: int | None = None,
    input_type: str = "codes",
    endpoint: str = "endpoints",
    exclude_endpoints: bool = True,
    create_plot: bool = True,
    show_title: bool = True,
    col_title: str | None = None,
    axes=None,
    ax=None,
) -> dict:
    """
    Analyze static INL/DNL from ADC output codes captured during a ramp test.

    This function is parallel to :func:`analyze_inl_from_sine`, but it uses the
    ramp-test assumption that integer ADC codes were collected while the input
    was swept by a monotonic linear ramp. The first implementation supports
    code histograms only; transition extraction from analog ramp/input-voltage
    pairs should be handled by a future API extension.

    By default this wrapper reports endpoint INL because
    ``endpoint='endpoints'`` forces the first and last analyzed INL samples to
    zero. Pass ``endpoint='fit'`` for best-fit-corrected INL, or
    ``endpoint='none'`` for raw cumulative INL and direct comparison with the
    current sine-histogram analyzer's baseline convention.

    Parameters
    ----------
    codes : array_like
        Integer ADC output codes from the ramp capture.
    num_bits : int, optional
        ADC resolution. Used to derive the full code range when ``code_max`` is
        omitted.
    code_min : int, default=0
        Lowest allowed ADC code.
    code_max : int, optional
        Highest allowed ADC code.
    input_type : {'codes'}, default='codes'
        Reserved for future voltage/transition-level support. Only ``'codes'``
        is currently implemented. Passing arbitrary non-ramp code streams will
        return histogram results, but they are not meaningful ramp DNL/INL.
    endpoint : {'endpoints', 'fit', 'none'}, default='endpoints'
        INL baseline correction passed to :func:`compute_inl_from_ramp`.
    exclude_endpoints : bool, default=True
        Exclude first and last codes from reported DNL/INL.
    create_plot : bool, default=True
        Plot DNL/INL using :func:`plot_dnl_inl`.
    show_title : bool, default=True
        Show auto-generated plot titles with min/max ranges.
    col_title : str, optional
        Optional title prefix for subplot layouts.
    axes : tuple of matplotlib.axes.Axes, optional
        Existing ``(dnl_ax, inl_ax)`` axes to draw into.
    ax : matplotlib.axes.Axes, optional
        Single axis to split into DNL/INL sub-axes.

    Returns
    -------
    dict
        Result dictionary from :func:`compute_inl_from_ramp`.
    """
    if input_type != "codes":
        raise ValueError("analyze_inl_from_ramp currently supports only input_type='codes'")

    result = compute_inl_from_ramp(
        codes=codes,
        num_bits=num_bits,
        code_min=code_min,
        code_max=code_max,
        endpoint=endpoint,
        exclude_endpoints=exclude_endpoints,
    )

    if create_plot:
        plot_dnl_inl(
            code=result["code"],
            dnl=result["dnl"],
            inl=result["inl"],
            num_bits=num_bits,
            show_title=show_title,
            col_title=col_title,
            axes=axes,
            ax=ax,
        )

    return result
