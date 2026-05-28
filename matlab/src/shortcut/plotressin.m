function plotressin(bits, varargin)
%PLOTRESSIN Plot partial-sum residuals using sine-wave calibration
%   Convenience wrapper that calibrates bit weights via WCALSIN and then
%   forwards the calibrated weights and reconstructed ideal signal to
%   PLOTRES, eliminating the manual step of running WCALSIN beforehand.
%
%   Syntax:
%     PLOTRESSIN(bits)
%     PLOTRESSIN(bits, xy)
%     PLOTRESSIN(bits, ..., 'Name', Value)
%
%   Inputs:
%     bits - Raw ADC output bit matrix
%       Matrix (N x M), where N is the number of samples and M is the
%       number of bits. Each column represents one bit (MSB first).
%
%     xy - Pairs of bit indices whose residuals are plotted (optional)
%       Same format as PLOTRES: vector (1x2), matrix (Px2), or preset string
%       Preset strings:
%         'sig': [zeros(M,1), (1:M)']
%         'res': [(0:(M-1))', ones(M,1)*M] (default)
%         'bit': [(0:M-1)', (1:M)']
%       Default: [(0:(M-1))', ones(M,1)*M]
%
%   Name-Value Arguments:
%     xy      - Bit-pair indices or preset string (same as optional xy)
%     freq    - Normalized input frequency (Fin/Fs). Default: 0 (auto)
%     order   - Number of harmonics in the fitting model. Default: 1
%     verbose - Verbose output flag. Default: 0
%     alpha   - Marker transparency forwarded to PLOTRES. Default: 'auto'
%
%   Examples:
%     % Basic residual plot (weights recovered automatically)
%     N = 1024; M = 6;
%     sig = (sin(2*pi*(0:N-1)'/N * 3)/2 + 0.5) * (2^M - 1);
%     code = round(sig);
%     bits = dec2bin(code, M) - '0';
%     plotressin(bits)
%
%     % Specific bit pairs
%     plotressin(bits, [2 4; 4 6])
%
%     % Preset bit-pair selections
%     plotressin(bits, 'sig')
%
%     % Forward calibration parameters
%     plotressin(bits, 'order', 3)
%
%     % Combine xy pairs with name-value arguments
%     plotressin(bits, [0 6; 3 6], 'freq', 3/1024)
%
%   Notes:
%     - Internally calls WCALSIN to obtain calibrated weights and the
%       best-fit ideal sinewave, then passes them to PLOTRES.
%     - The reconstructed reference signal is ideal + offset (DC restored).
%
%   See also: plotres, wcalsin, plotwgt

    [N, M] = size(bits);
    if N < M
        bits = bits';
        M = size(bits, 2);
    end

    [xy, freq, order, verbose, alpha] = parsePlotressinInputs(varargin, M);

    % Calibrate weights and recover ideal sinewave
    [weight, offset, ~, ideal] = wcalsin(bits, ...
        'freq', freq, 'order', order, 'verbose', verbose);

    % Reconstruct the full reference signal (DC restored)
    sig = ideal + offset;

    % Plot residuals
    plotres(sig, bits, weight, xy, 'alpha', alpha);

end

function [xy, freq, order, verbose, alpha] = parsePlotressinInputs(args, M)
    xy = resolvePlotressinXy('res', M);
    freq = 0;
    order = 1;
    verbose = 0;
    alpha = 'auto';

    idx = 1;
    numArgs = numel(args);

    if idx <= numArgs
        if isPlotressinTextScalar(args{idx})
            if isPlotressinXyPreset(args{idx})
                xy = resolvePlotressinXy(args{idx}, M);
                idx = idx + 1;
            elseif ~isPlotressinParameterName(args{idx})
                error('plotressin:invalidInput', 'Invalid xy preset "%s". Use ''sig'', ''res'', or ''bit''.', char(args{idx}));
            end
        else
            validatePlotressinXy(args{idx});
            xy = resolvePlotressinXy(args{idx}, M);
            idx = idx + 1;
        end
    end

    while idx <= numArgs
        if ~isPlotressinTextScalar(args{idx})
            error('plotressin:invalidInput', 'Name-value argument names must be text.');
        end
        if ~isPlotressinParameterName(args{idx})
            error('plotressin:invalidInput', 'Unrecognized parameter name "%s".', char(args{idx}));
        end

        name = lower(char(args{idx}));
        idx = idx + 1;

        if idx > numArgs
            error('plotressin:invalidInput', 'Missing value for parameter "%s".', name);
        end

        switch name
            case 'xy'
                validatePlotressinXy(args{idx});
                xy = resolvePlotressinXy(args{idx}, M);
            case 'freq'
                freq = args{idx};
            case 'order'
                order = args{idx};
            case 'verbose'
                verbose = args{idx};
            case 'alpha'
                alpha = args{idx};
        end
        idx = idx + 1;
    end
end

function validatePlotressinXy(xy)
    if isPlotressinTextScalar(xy)
        return;
    end

    if ~(isnumeric(xy) && ismatrix(xy) && (size(xy, 1) == 2 || size(xy, 2) == 2))
        error('plotressin:invalidInput', 'xy must be numeric bit-pair indices or one of ''sig'', ''res'', or ''bit''.');
    end
end

function xy = resolvePlotressinXy(xy, M)
    if ~isPlotressinTextScalar(xy)
        return;
    end

    preset = char(xy);
    switch lower(preset)
        case 'sig'
            xy = [zeros(M, 1), (1:M)'];
        case 'res'
            xy = [(0:(M-1))', ones(M, 1) * M];
        case 'bit'
            xy = [(0:M-1)', (1:M)'];
        otherwise
            error('plotressin:invalidInput', 'Invalid xy preset "%s". Use ''sig'', ''res'', or ''bit''.', preset);
    end
end

function tf = isPlotressinTextScalar(x)
    tf = (ischar(x) && isrow(x)) || (isstring(x) && isscalar(x));
end

function tf = isPlotressinXyPreset(x)
    tf = isPlotressinTextScalar(x) && any(strcmpi(char(x), {'sig', 'res', 'bit'}));
end

function tf = isPlotressinParameterName(x)
    tf = isPlotressinTextScalar(x) && any(strcmpi(char(x), {'xy', 'freq', 'order', 'verbose', 'alpha'}));
end
