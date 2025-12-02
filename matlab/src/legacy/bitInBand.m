function [signalOut] = bitInBand(signalIn, bands)
%BITINBAND Filter signal to retain only specified frequency bands
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use IFILTER instead.
%
%   This function is a wrapper that calls ifilter with the same
%   functionality. All new code should use ifilter directly.
%
%   Syntax:
%     signalOut = BITINBAND(signalIn, bands)
%
%   Inputs:
%     signalIn - Input signal matrix (must be real-valued)
%     bands - Frequency band specifications (normalized to sampling frequency)
%
%   Outputs:
%     signalOut - Filtered signal with only in-band components retained
%
%   See also: ifilter, alias, fft, ifft

    % Call the new ifilter function
    signalOut = ifilter(signalIn, bands);

end
