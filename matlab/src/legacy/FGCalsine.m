function [weight,offset,postcal,ideal,err,freqcal] = FGCalsine(bits,varargin)
%FGCALSINE Foreground calibration using a sine wave input (legacy)
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use wcalsine (lowercase) instead.
%
%   This function is a wrapper that calls wcalsine with the same
%   functionality. All new code should use wcalsine directly.
%
%   Legacy interface:
%     [weight, offset, postcal, ideal, err, freqcal] = FGCALSINE(bits)
%     [weight, offset, postcal, ideal, err, freqcal] = FGCALSINE(bits, Name, Value)
%
%   Inputs:
%     bits - Binary ADC output data
%       Same meaning in the new function
%
%   Name-Value Arguments:
%     freq - Normalized input frequency
%       Same meaning in the new function
%     rate - Adaptive rate for frequency updates
%       Same meaning in the new function
%     reltol - Relative error tolerance
%       Same meaning in the new function
%     niter - Maximum iterations for fine frequency search
%       Same meaning in the new function
%     order - Number of harmonics to exclude from fit
%       Same meaning in the new function
%     fsearch - Force fine frequency search
%       Same meaning in the new function
%     nomWeight - Nominal bit weights
%       Same meaning in the new function
%
%   Outputs:
%     weight - Calibrated bit weights
%       Same meaning in the new function
%     offset - Calibrated DC offset
%       Same meaning in the new function
%     postcal - Signal after weight calibration
%       Same meaning in the new function
%     ideal - Best-fit sine wave
%       Same meaning in the new function
%     err - Residual error after calibration
%       Same meaning in the new function
%     freqcal - Fine-tuned normalized frequency
%       Same meaning in the new function
%
%   See also: wcalsine

    % Call the new wcalsine function with all arguments
    [weight,offset,postcal,ideal,err,freqcal] = wcalsine(bits,varargin{:});

end
