function [data_fit,freq,mag,dc,phi] = sineFit(data,f0,tol,rate)
%SINEFIT Four-parameter iterative sine wave fitting (legacy)
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use sinfit (lowercase) instead.
%
%   This function is a wrapper that calls sinfit with the same
%   functionality. All new code should use sinfit directly.
%
%   Legacy interface:
%     [data_fit, freq, mag, dc, phi] = SINEFIT(data, f0, tol, rate)
%
%   Inputs:
%     data - Input signal to be fitted
%     f0 - Initial frequency estimate (optional)
%     tol - Convergence tolerance (optional)
%     rate - Step size rate for frequency update (optional)
%
%   Outputs:
%     data_fit - Fitted sine wave signal
%     freq - Fitted normalized frequency
%     mag - Fitted signal amplitude
%     dc - Fitted DC offset
%     phi - Fitted phase in radians
%
%   See also: sinfit

    % Call the new sinfit function with appropriate arguments
    if nargin == 1
        [data_fit,freq,mag,dc,phi] = sinfit(data);
    elseif nargin == 2
        [data_fit,freq,mag,dc,phi] = sinfit(data,f0);
    elseif nargin == 3
        [data_fit,freq,mag,dc,phi] = sinfit(data,f0,tol);
    else
        [data_fit,freq,mag,dc,phi] = sinfit(data,f0,tol,rate);
    end

end
