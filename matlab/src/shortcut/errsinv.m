function [emean, erms, xx, anoi, pnoi, err, errxx] = errsinv(sig, varargin)
%ERRSINV Shortcut for errsin with xaxis='value' (bin by signal value)
%   Wrapper for ERRSIN that defaults xaxis to 'value' mode.
%
%   See also: errsin

    if nargout == 0
        errsin(sig, 'xaxis', 'value', 'disp', 1, varargin{:});
    else
        [emean, erms, xx, anoi, pnoi, err, errxx] = errsin(sig, 'xaxis', 'value', varargin{:});
    end

end