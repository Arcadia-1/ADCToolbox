"""Helpers for exposing sine-fit diagnostics from AOUT analyses."""

FIT_DIAGNOSTIC_KEYS = (
    "frequency",
    "amplitude",
    "phase",
    "dc_offset",
    "rmse",
    "converged",
    "n_iterations",
    "initial_frequency",
    "last_delta_freq",
)


def extract_fit_diagnostics(fit_result):
    """Return fit metadata without duplicating fitted waveform arrays."""
    return {key: fit_result[key] for key in FIT_DIAGNOSTIC_KEYS}
