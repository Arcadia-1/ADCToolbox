function [INL, DNL, code] = INLsine(data, clip)
%INLSINE Calculate INL and DNL from sine wave histogram test (legacy)
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use inlsin (lowercase) instead.
%
%   This function is a wrapper that calls inlsin with the same
%   functionality. All new code should use inlsin directly.
%
%   Legacy interface:
%     [INL, DNL, code] = INLSINE(data, clip)
%
%   Inputs:
%     data - ADC output codes from sine wave input
%     clip - Exclusion ratio for endpoints (optional, default: 0.01)
%       Now named 'excl' in the new function
%
%   Outputs:
%     INL - Integral Nonlinearity in LSB
%       Now named 'inl' in the new function
%     DNL - Differential Nonlinearity in LSB
%       Now named 'dnl' in the new function
%     code - Code values corresponding to INL/DNL measurements
%
%   Notes:
%     - The new inlsin function has additional display functionality
%     - This wrapper disables auto-display to maintain legacy behavior
%
%   See also: inlsin

    % Call the new inlsin function with appropriate arguments
    % Always disable display to maintain legacy behavior
    if nargin == 1
        [INL, DNL, code] = inlsin(data, 0.01, false);
    else
        [INL, DNL, code] = inlsin(data, clip, false);
    end

end
