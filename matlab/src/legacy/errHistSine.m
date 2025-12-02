function [emean, erms, phase_code, anoi, pnoi, err, xx] = errHistSine(data, varargin)
%ERRHISTSINE Analyze sinewave fit errors with histogram binning (legacy)
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use errsin (lowercase) instead.
%
%   This function is a wrapper that calls errsin with the same
%   functionality. All new code should use errsin directly.
%
%   Legacy interface:
%     [emean, erms, phase_code, anoi, pnoi, err, xx] = ERRHISTSINE(data)
%     [emean, erms, phase_code, anoi, pnoi, err, xx] = ERRHISTSINE(data, Name, Value)
%
%   Inputs:
%     data - Input signal data
%       Maps to 'sig' in the new function
%
%   Name-Value Arguments:
%     'bin' - Number of bins for histogram analysis
%       Same meaning in the new function
%     'fin' - Normalized input frequency
%       Same meaning in the new function
%     'disp' - Display plots
%       Same meaning in the new function
%     'mode' - Analysis mode (0=phase, >=1=value)
%       Maps to 'xaxis' in the new function ('phase' or 'value')
%     'erange' - Error range filter
%       Same meaning in the new function
%
%   Outputs:
%     emean - Mean error for each bin
%       Same meaning in the new function
%     erms - RMS error for each bin
%       Same meaning in the new function
%     phase_code - Bin centers (phase or value)
%       Maps to 'xx' in the new function
%     anoi - Estimated amplitude noise RMS
%       Same meaning in the new function
%     pnoi - Estimated phase noise RMS
%       Same meaning in the new function
%     err - Raw errors for each sample point
%       Same meaning in the new function
%     xx - X-axis values corresponding to raw errors
%       Maps to 'errxx' in the new function
%
%   See also: errsin

    % Convert legacy 'mode' parameter to new 'xaxis' parameter
    % Parse and convert legacy arguments to new format
    newArgs = {};
    ii = 1;
    while ii <= length(varargin)
        if ischar(varargin{ii}) && strcmpi(varargin{ii}, 'mode')
            % Convert mode to xaxis
            if ii+1 <= length(varargin)
                modeVal = varargin{ii+1};
                if modeVal == 0
                    newArgs{end+1} = 'xaxis';
                    newArgs{end+1} = 'phase';
                else
                    newArgs{end+1} = 'xaxis';
                    newArgs{end+1} = 'value';
                end
                ii = ii + 2;
            else
                error('errHistSine:invalidMode', 'Mode parameter requires a value.');
            end
        else
            % Pass through other arguments
            newArgs{end+1} = varargin{ii};
            ii = ii + 1;
        end
    end

    % Call the new errsin function with converted arguments
    % Note: errsin has output order [emean, erms, xx, anoi, pnoi, err, errxx]
    % Legacy has output order [emean, erms, phase_code, anoi, pnoi, err, xx]
    % So we swap outputs 3 and 7
    [emean, erms, phase_code, anoi, pnoi, err, xx] = errsin(data, newArgs{:});

end