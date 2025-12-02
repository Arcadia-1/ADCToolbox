function freq = findfreq(sig,fs)
%FINDFREQ Find dominant frequency in signal using sine wave fitting
%   This function identifies the dominant frequency component in a signal by
%   performing four-parameter iterative sine wave fitting. It extracts the
%   normalized frequency and scales it by the sampling frequency to return
%   the actual frequency.
%
%   Syntax:
%     freq = FINDFREQ(sig, fs)
%
%   Inputs:
%     sig - Input signal to analyze. Must be real.
%       Vector
%     fs - Sampling frequency. Must be a positive real number.
%       Scalar
%       Default: 1 (normalized frequency)
%
%   Outputs:
%     freq - Dominant frequency of the signal
%       Scalar
%       Range: [0, fs/2] (Nyquist limit)
%
%   Examples:
%     % Find frequency of a 1 kHz sine wave sampled at 10 kHz
%     t = 0:1/10000:1;
%     sig = sin(2*pi*1000*t);
%     freq = findfreq(sig, 10000)  % Returns ~1000
%
%     % Find normalized frequency (fs = 1)
%     freq = findfreq(sig)  % Returns ~0.1
%
%   Notes:
%     - Uses iterative sine fitting algorithm (sinefit) internally
%     - Returns the fitted frequency, not FFT peak frequency
%     - Signal should contain a dominant sinusoidal component
%     - For multi-tone signals, returns frequency of largest component
%
%   See also: sinefit, alias, findbin

    % Handle default sampling frequency
    if(nargin < 2)
        fs = 1;
    end

    % Input validation
    if fs <= 0
        error('findfreq:invalidFs', 'Sampling frequency fs must be positive.');
    end

    if(~isreal(sig) || ~isreal(fs))
        error('findfreq:invalidInput', 'Inputs must be real numbers.');
    end

    % Fit sine wave to extract normalized frequency
    [~,freq,~,~,~] = sinefit(sig);

    % Scale by sampling frequency to get actual frequency in Hz
    freq = freq*fs;

end