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
%       Same format as PLOTRES: vector (1x2) or matrix (Px2)
%       Default: [(0:(M-1))', ones(M,1)*M]
%
%   Name-Value Arguments (forwarded to WCALSIN):
%     freq    - Normalized input frequency (Fin/Fs). Default: 0 (auto)
%     order   - Number of harmonics in the fitting model. Default: 1
%     verbose - Verbose output flag. Default: 0
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
        [N, M] = size(bits);
    end

    p = inputParser;
    addOptional(p, 'xy', [(0:(M-1))', ones(M,1)*M], ...
        @(x) isnumeric(x) && ismatrix(x) && (size(x,1)==2 || size(x,2)==2));
    addParameter(p, 'freq', 0);
    addParameter(p, 'order', 1);
    addParameter(p, 'verbose', 0);
    addParameter(p, 'alpha', 'auto');
    parse(p, varargin{:});

    xy      = p.Results.xy;
    freq    = p.Results.freq;
    order   = p.Results.order;
    verbose = p.Results.verbose;
    alpha   = p.Results.alpha;

    % Calibrate weights and recover ideal sinewave
    [weight, offset, ~, ideal] = wcalsin(bits, ...
        'freq', freq, 'order', order, 'verbose', verbose);

    % Reconstruct the full reference signal (DC restored)
    sig = ideal + offset;

    % Plot residuals
    plotres(sig, bits, weight, xy, 'alpha', alpha);

end
