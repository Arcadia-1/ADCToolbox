import numpy as np

def inl_sine(data, clip=0.01):
    """
    Calculate ADC INL (Integral Nonlinearity) and DNL (Differential Nonlinearity) using sine histogram method

    This function estimates INL and DNL by applying an arccosine transform to the histogram
    of ADC output data. The basic principle: for an ideal sinusoidal input, the cumulative
    distribution function (CDF) of ADC output codes should follow an arcsine function.
    By applying an arccosine transform to the cumulative histogram, we obtain the ideal
    linear code distribution, allowing calculation of nonlinearity errors.

    Parameters
    ----------
    data : ndarray
        ADC output data (1D array)
    clip : float, optional
        Clipping ratio to exclude outliers at data edges, default is 0.01 (1%)

    Returns
    -------
    INL : ndarray
        Integral nonlinearity error (in LSB units)
    DNL : ndarray
        Differential nonlinearity error (in LSB units)
    code : ndarray
        Corresponding output code values

    Notes
    -----
    INL represents the maximum deviation of the actual transfer characteristic from the ideal line
    DNL represents the deviation of adjacent code step sizes from the ideal step size (1 LSB)
    """
    
    # Ensure data is a column vector (transpose if needed)
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    S = data.shape
    if S[0] < S[1]:  # If rows < columns, transpose
        data = data.T

    # Flatten to 1D array
    data = data.flatten()

    # Calculate data range and apply clipping
    # Clipping excludes data points in the saturation region, which cause statistical bias
    max_data = np.ceil(np.max(data))
    min_data = np.floor(np.min(data))

    # Clip both ends of the data range by clip ratio
    max_data = round(max_data - clip * (max_data - min_data) / 2)
    min_data = round(min_data + clip * (max_data - min_data) / 2)

    # Generate code value range
    code = np.arange(min_data, max_data + 1)

    # Restrict data to clipped range
    data = np.clip(data, min_data, max_data)

    # Calculate histogram - count occurrences of each code
    # DCC: Digital Code Count, count for each code
    # MATLAB's hist(data, code) uses code values as bin centers
    # For consecutive integers, bins are [code[i]-0.5, code[i]+0.5)
    bins = np.append(code - 0.5, code[-1] + 0.5)
    DCC, _ = np.histogram(data, bins=bins)

    # Calculate cumulative distribution function (CDF) and apply arccosine transform
    # Theory: for ideal sine wave, CDF should follow arcsine function
    # Use -cos(π * CDF) transform to linearize it
    # cumsum(DCC)/sum(DCC) calculates normalized CDF (range 0 to 1)
    cumulative_prob = np.cumsum(DCC) / np.sum(DCC)
    DCC = -np.cos(np.pi * cumulative_prob)

    # Calculate differential: difference between adjacent codes
    # DNL is the deviation of each code step size from ideal 1 LSB
    DNL = DCC[1:] - DCC[:-1]

    # Update code range (differential reduces by one point)
    code = code[:-1]

    # Apply edge clipping again to DNL and code values
    # This excludes edge effects, as statistics at edges may be inaccurate
    clip_points = int(np.floor(clip * (max_data - min_data + 1) / 2))

    if clip_points > 0:
        code = code[clip_points:-clip_points]
        DNL = DNL[clip_points:-clip_points]

    # DNL normalization processing
    # 1. First normalize so sum equals 1 (probability normalization)
    DNL = DNL / np.sum(DNL)

    # 2. Scale to actual code range (in LSB units)
    # Ideally, each code step size should be 1 LSB
    num_codes = max_data - min_data - clip_points * 2 + 1
    DNL = DNL * num_codes - 1

    # 3. Remove mean (removes gain error influence)
    # This way DNL reflects only nonlinearity, excluding overall gain error
    DNL = DNL - np.mean(DNL)

    # Calculate INL: cumulative sum of DNL
    # INL represents deviation of each code point from ideal line
    # Obtained by integrating (accumulating) DNL
    INL = np.cumsum(DNL)

    return INL, DNL, code


