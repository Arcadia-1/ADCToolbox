def alias(Fin, Fs):
    """
    Calculate aliased frequency.

    Args:
        Fin: Input frequency (Hz)
        Fs: Sampling rate (Hz)

    Returns:
        Aliased frequency in baseband [0, Fs/2] (Hz)
    """
    cycle = int(Fin / Fs * 2)
    base_freq = Fin - int(Fin / Fs) * Fs

    if cycle % 2 != 0:  # Odd zone: reflected
        f_alias = Fs - base_freq
    else:  # Even zone: direct
        f_alias = base_freq

    return f_alias
