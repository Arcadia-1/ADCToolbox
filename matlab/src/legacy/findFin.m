function fin = findFin(data,Fs)
%FINDFIN Find dominant frequency in signal (legacy)
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use findfreq (lowercase) instead.
%
%   This function is a wrapper that calls findfreq with the same
%   functionality. All new code should use findfreq directly.
%
%   Legacy interface:
%     fin = FINDFIN(data, Fs)
%
%   Inputs:
%     data - Input signal to analyze
%     Fs - Sampling frequency (optional, default: 1)
%
%   Outputs:
%     fin - Dominant frequency of the signal in Hz
%
%   See also: findfreq

    % Call the new findfreq function with appropriate arguments
    if nargin < 2
        fin = findfreq(data);
    else
        fin = findfreq(data,Fs);
    end

end
