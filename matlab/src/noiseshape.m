function [sigout, info] = noiseshape(varargin)
%NOISESHAPE Generate noise-shaped quantized ADC-like output.
%   This helper creates a signal with quantization error shaped by an NTF.
%   It can either process an existing input waveform or generate a sinewave
%   internally before applying quantization and noise shaping.
%
%   Syntax:
%     sigout = noiseshape(sigin)
%     sigout = noiseshape(sigin, 'Name', Value)
%     sigout = noiseshape('N', 8192, 'Fs', 100e6, 'Fin', 1e6)
%     [sigout, info] = noiseshape(...)
%
%   Name-Value Arguments:
%     'N' - Number of samples when no input signal is provided
%       Positive integer, default: 8192
%     'Fs' - Sampling frequency used for generated sinewave
%       Positive scalar, default: 1
%     'Fin' - Sinewave frequency used for generated sinewave
%       Positive scalar, default: 1/64
%     'A' - Sinewave peak amplitude
%       Scalar, default: 0.4
%     'DC' - Sinewave DC offset
%       Scalar, default: 0
%     'bits' - Quantizer resolution
%       Positive integer, default: 10
%     'range' - Quantizer input range [min max]
%       Two-element increasing vector, default: [-0.5 0.5]
%     'order' - Lowpass NTF order for default NTF
%       Integer from 1 to 5, default: 1
%     'ntf' - Custom NTF
%       [] for default (1-z^-1)^order; vector numerator; {num, den};
%       struct with fields .num and optional .den; or Control System Toolbox
%       tf/zpk/ss object when tfdata is available.
%
%   Outputs:
%     sigout - Signal plus noise-shaped quantization error
%     info - Structure with clean signal, quantized signal, quantization
%       error, shaped error, NTF numerator/denominator, and codes.
%
%   Examples:
%     % Generate a 2nd-order lowpass noise-shaped output
%     sig = noiseshape('N', 2^13, 'Fs', 100e6, 'Fin', 0.5e6, ...
%                      'bits', 10, 'order', 2);
%     plotspec(sig, 100e6, 1, 'OSR', 32);
%
%     % Use a custom FIR NTF numerator
%     sig = noiseshape(x, 'ntf', [1 -1.6 0.64], 'bits', 10);
%
%   Notes:
%     - This is a lightweight signal generator for examples and testbenches.
%       It is not a full delta-sigma loop simulator.
%     - The default NTF shapes the quantization error with
%       (1 - z^-1)^order, matching the Python ADC_Signal_Generator helper.
%
%   See also: ntfperf, plotspec, perfosr, ifilter

    if nargin > 0 && ~(ischar(varargin{1}) || isstring(varargin{1}))
        sigin = varargin{1};
        args = varargin(2:end);
    else
        sigin = [];
        args = varargin;
    end

    p = inputParser;
    addParameter(p, 'N', 8192, @(x) isnumeric(x) && isscalar(x) && x > 0 && mod(x,1) == 0);
    addParameter(p, 'Fs', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'Fin', 1/64, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'A', 0.4, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'DC', 0, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'bits', 10, @(x) isnumeric(x) && isscalar(x) && x > 0 && mod(x,1) == 0);
    addParameter(p, 'range', [-0.5 0.5], @(x) isnumeric(x) && numel(x) == 2 && x(2) > x(1));
    addParameter(p, 'order', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1 && x <= 5 && mod(x,1) == 0);
    addParameter(p, 'ntf', []);
    parse(p, args{:});

    if isempty(sigin)
        n = p.Results.N;
        t = (0:n-1)' / p.Results.Fs;
        sigin = p.Results.A * sin(2*pi*p.Results.Fin*t) + p.Results.DC;
    else
        if ~isreal(sigin)
            error('noiseshape:complexInput', 'Input signal must be real-valued.');
        end
        if isrow(sigin)
            sigin = sigin';
        end
        n = numel(sigin);
    end

    [num, den] = resolveNtf(p.Results.ntf, p.Results.order);

    qrange = p.Results.range(:)';
    vmin = qrange(1);
    vmax = qrange(2);
    lsb = (vmax - vmin) / 2^p.Results.bits;
    maxCode = 2^p.Results.bits - 1;

    codes = floor((sigin - vmin) / lsb);
    codes = min(max(codes, 0), maxCode);
    sigq = codes * lsb + vmin;

    qerr = sigq - sigin;
    shapedErr = filter(num, den, qerr);
    sigout = sigin + shapedErr;

    if nargout > 1
        info = struct();
        info.clean = sigin;
        info.quantized = sigq;
        info.quantization_error = qerr;
        info.shaped_error = shapedErr;
        info.codes = codes;
        info.ntf_num = num;
        info.ntf_den = den;
        info.lsb = lsb;
        info.N = n;
    end
end

function [num, den] = resolveNtf(ntf, order)
    if isempty(ntf)
        k = 0:order;
        num = (-1).^k .* arrayfun(@(x) nchoosek(order, x), k);
        den = 1;
        return;
    end

    if isnumeric(ntf)
        num = ntf(:)';
        den = 1;
        return;
    end

    if iscell(ntf)
        if numel(ntf) ~= 2
            error('noiseshape:invalidNtf', 'Cell NTF must be {num, den}.');
        end
        num = ntf{1}(:)';
        den = ntf{2}(:)';
        return;
    end

    if isstruct(ntf)
        if ~isfield(ntf, 'num')
            error('noiseshape:invalidNtf', 'Struct NTF must contain field num.');
        end
        num = ntf.num(:)';
        if isfield(ntf, 'den')
            den = ntf.den(:)';
        else
            den = 1;
        end
        return;
    end

    if exist('tfdata', 'file') == 2
        try
            [numCell, denCell] = tfdata(ntf, 'v');
            num = numCell(:)';
            den = denCell(:)';
            return;
        catch
            % Fall through to error below.
        end
    end

    error('noiseshape:invalidNtf', 'Unsupported NTF format.');
end

