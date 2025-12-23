function [emean, erms, xx, anoi, pnoi, err, errxx] = errsinv(sig, varargin)
%ERRSINV Shortcut for errsin with xaxis='value' (bin by signal value)
%   Wrapper for ERRSIN that defaults xaxis to 'value' mode.
%
%   See also: errsin

    [emean, erms, xx, anoi, pnoi, err, errxx] = errsin(sig, 'xaxis', 'value', varargin{:});

end