function plotres(sig, bits, varargin)
%PLOTRES Plot partial-sum residuals of an ADC bit matrix
%   Visualizes the relationship between residuals at different bit stages
%   of an ADC. For each specified pair (x, y), the function computes the
%   partial-sum residual after subtracting the first x (or y) weighted bits
%   from the input signal, then plots the y-residual vs. the x-residual as
%   a scatter plot. This reveals correlations, nonlinearity patterns, and
%   redundancy between bit stages.
%
%   Syntax:
%     PLOTRES(sig, bits)
%     PLOTRES(sig, bits, xyPreset)
%     PLOTRES(sig, bits, wgt)
%     PLOTRES(sig, bits, wgt, xy)
%     PLOTRES(sig, bits, wgt, xy, alpha)
%     PLOTRES(sig, bits, 'Name', Value)
%
%   Inputs:
%     sig - Ideal input signal to the ADC
%       Vector (N x 1 or 1 x N), where N is the number of samples
%
%     bits - Raw ADC output bit matrix
%       Matrix (N x M), where M is the number of bits
%       Each column represents one bit of the ADC output (MSB first)
%       Code format: [MSB, MSB-1, ..., LSB]
%
%     wgt - Bit weights for ADC code calculation (optional)
%       Vector (1 x M)
%       Default: binary weights [2^(M-1), 2^(M-2), ..., 2, 1]
%
%     xy - Pairs of bit indices whose residuals are plotted (optional)
%       Vector (1 x 2 or 2 x 1) or matrix (P x 2 or 2 x P)
%       Format: [x_bit, y_bit] or [x1,y1; x2,y2; ...]
%       Range of x_bit and y_bit: [0, M]
%         0 means the raw input signal (no bits subtracted)
%         1..M means the residual after subtracting the first 1..M bits
%       Preset strings:
%         'sig': [zeros(M,1), (1:M)']
%         'res': [(0:(M-1))', ones(M,1)*M] (default)
%         'bit': [(0:M-1)', (1:M)']
%       Default: [0, M; 1, M; ...; M-1, M]
%
%     alpha - Marker transparency for scatter points
%       'auto' (default): scales alpha with sample count N as
%         clamp(1000/N, 0.1, 1)
%       Numeric scalar in (0, 1]: fixed transparency value
%
%   Examples:
%     % Basic residual plot with binary weights
%     N = 1024; M = 6;
%     sig = (sin(2*pi*(0:N-1)'/N * 3)/2 + 0.5) * (2^M - 1);
%     code = round(sig);
%     bits = dec2bin(code, M) - '0';
%     plotres(sig, bits);
%
%     % Preset bit-pair selections
%     plotres(sig, bits, 'bit');
%
%     % Custom weights
%     wgt = 2.^(M-1:-1:0);
%     plotres(sig, bits, wgt);
%
%     % Specific bit pairs
%     plotres(sig, bits, wgt, [2 4; 4 6]);
%
%   Notes:
%     - Each subplot shows residual_y vs. residual_x as a scatter plot
%     - Residual at stage k = sig - bits(:,1:k) * wgt(1:k)'
%     - Stage 0 residual is the raw signal itself
%     - Patterns in the scatter reveal inter-stage correlation or nonlinearity
%
%   See also: bitchk, plotwgt

    [N,M] = size(bits);
    [Ns,Ms] = size(sig);
    if(Ms == 1)
        if(Ns ~= N)
            error('plotres:invalidInput', 'The number of samples of sig and bits must be the same.');
        end
    else
        if(Ns ~= 1)
            error('plotres:invalidInput', 'sig must be a vector.');
        end
        if(Ms ~= N)
            error('plotres:invalidInput', 'The number of samples of sig and bits must be the same.');
        end
        sig = sig';
    end

    [wgt, xy, alphaVal] = parsePlotresInputs(varargin, M);

    % Resolve alpha
    if ischar(alphaVal) || isstring(alphaVal)
        alphaVal = min(max(1000 / N, 0.1), 1);
    end

    [Nxy,Mxy] = size(xy);
    if(Mxy == 2)
        P = Nxy;
    else
        P = Mxy;
        xy = xy';
    end

    figure;
    tiledlayout('flow');

    for k = 1:P

        nexttile;

        res_x = sig - bits(:,1:xy(k,1))*wgt(1:xy(k,1))';
        res_y = sig - bits(:,1:xy(k,2))*wgt(1:xy(k,2))';

        scatter(res_x, res_y, 4, 'k', 'filled', 'MarkerFaceAlpha', alphaVal);

        grid on;

        if(xy(k,1) == 0)
            xlabel('Signal');
        else
            xlabel(sprintf('Res. of bit #%d', xy(k,1)));
        end
        if(xy(k,2) == 0)
            ylabel('Signal');
        else
            ylabel(sprintf('Res. of bit #%d', xy(k,2)));
        end

    end

end

function [wgt, xy, alphaVal] = parsePlotresInputs(args, M)
    wgt = 2.^(M-1:-1:0);
    xy = resolvePlotresXy('res', M);
    alphaVal = 'auto';

    idx = 1;
    numArgs = numel(args);

    if idx <= numArgs
        if isPlotresTextScalar(args{idx})
            if ~isPlotresParameterName(args{idx})
                xy = resolvePlotresXy(args{idx}, M);
                idx = idx + 1;
                if idx <= numArgs && ~isPlotresParameterName(args{idx})
                    if idx == numArgs
                        [alphaVal, idx] = readPlotresAlpha(args, idx);
                    else
                        error('plotres:invalidInput', 'Too many input arguments.');
                    end
                end
            end
        else
            validatePlotresWeight(args{idx});
            wgt = args{idx};
            idx = idx + 1;

            if idx <= numArgs
                if ~isPlotresParameterName(args{idx})
                    validatePlotresXy(args{idx});
                    xy = resolvePlotresXy(args{idx}, M);
                    idx = idx + 1;
                    if idx <= numArgs && ~isPlotresParameterName(args{idx})
                        if idx == numArgs
                            [alphaVal, idx] = readPlotresAlpha(args, idx);
                        else
                            error('plotres:invalidInput', 'Too many input arguments.');
                        end
                    end
                end
            end
        end
    end

    while idx <= numArgs
        if ~isPlotresTextScalar(args{idx})
            error('plotres:invalidInput', 'Name-value argument names must be text.');
        end
        if ~isPlotresParameterName(args{idx})
            error('plotres:invalidInput', 'Unrecognized parameter name "%s".', char(args{idx}));
        end

        name = lower(char(args{idx}));
        idx = idx + 1;

        if idx > numArgs
            error('plotres:invalidInput', 'Missing value for parameter "%s".', name);
        end

        switch name
            case 'wgt'
                validatePlotresWeight(args{idx});
                wgt = args{idx};
                idx = idx + 1;
            case 'xy'
                validatePlotresXy(args{idx});
                xy = resolvePlotresXy(args{idx}, M);
                idx = idx + 1;
            case 'alpha'
                [alphaVal, idx] = readPlotresAlpha(args, idx);
        end
    end
end

function [alphaVal, idx] = readPlotresAlpha(args, idx)
    if idx > numel(args)
        error('plotres:invalidInput', 'Missing value for alpha.');
    end

    alphaVal = args{idx};
    if ~isValidPlotresAlpha(alphaVal)
        error('plotres:invalidInput', 'alpha must be ''auto'' or a numeric scalar in (0, 1].');
    end
    idx = idx + 1;
end

function validatePlotresWeight(wgt)
    if ~(isnumeric(wgt) && isvector(wgt))
        error('plotres:invalidInput', 'wgt must be a numeric vector.');
    end
end

function validatePlotresXy(xy)
    if isPlotresTextScalar(xy)
        return;
    end

    if ~(isnumeric(xy) && ismatrix(xy) && (size(xy, 1) == 2 || size(xy, 2) == 2))
        error('plotres:invalidInput', 'xy must be numeric bit-pair indices or one of ''sig'', ''res'', or ''bit''.');
    end
end

function xy = resolvePlotresXy(xy, M)
    if ~isPlotresTextScalar(xy)
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
            error('plotres:invalidInput', 'Invalid xy preset "%s". Use ''sig'', ''res'', or ''bit''.', preset);
    end
end

function tf = isPlotresTextScalar(x)
    tf = (ischar(x) && isrow(x)) || (isstring(x) && isscalar(x));
end

function tf = isPlotresParameterName(x)
    tf = isPlotresTextScalar(x) && any(strcmpi(char(x), {'wgt', 'xy', 'alpha'}));
end

function tf = isValidPlotresAlpha(x)
    tf = (ischar(x) && isrow(x) && strcmpi(x, 'auto')) || ...
        (isstring(x) && isscalar(x) && strcmpi(x, 'auto')) || ...
        (isnumeric(x) && isscalar(x) && x > 0 && x <= 1);
end
