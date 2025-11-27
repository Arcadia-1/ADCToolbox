function [signal, error, indep, dep, phi] = tomDecomp(data, re_fin, order, disp)
%TOMDECOMP Thompson decomposition for ADC signal analysis (legacy)
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use tomdec (lowercase) instead.
%
%   This function is a wrapper that calls tomdec with the same
%   functionality. All new code should use tomdec directly.
%
%   Legacy interface:
%     [signal, error, indep, dep, phi] = TOMDECOMP(data, re_fin, order, disp)
%
%   Inputs:
%     data - ADC data to be decomposed, 1xN vector
%     re_fin - relative input frequency (f_in/f_sample)
%     order - order of harmonics to be counted as dependent error
%     disp - turn on result display
%
%   Outputs:
%     signal - Reconstructed fundamental signal
%     error - Total error (data - signal)
%     indep - Independent error component
%     dep - Dependent error component (harmonics)
%     phi - Phase of the fundamental
%
%   See also: tomdec

    % Call the new tomdec function with appropriate arguments
    if nargin == 1
        [signal, error, dep, indep] = tomdec(data);
    elseif nargin == 2
        [signal, error, dep, indep] = tomdec(data, re_fin);
    elseif nargin == 3
        [signal, error, dep, indep] = tomdec(data, re_fin, order);
    else
        [signal, error, dep, indep] = tomdec(data, re_fin, order, disp);
    end

    if(nargout == 5)
        warning('TOMDECOMP: Deprecated output phi is no longer supported (phi = 0). Use tomdec instead.');
        phi = 0;
    end

end
