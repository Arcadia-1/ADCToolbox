function [sine, err, har, oth] = tomdec(sig, varargin)
%TOMDEC Thompson decomposition of a single-tone signal into sinewave and errors
%   This function performs Thompson decomposition on a single-tone signal, 
%   separating the signal into fundamental sinewave, harmonic distortions, and other
%   errors. The method uses least-squares fitting to decompose the signal.
%
%   Syntax:
%     [sine, err, har, oth] = TOMDEC(sig)
%     [sine, err, har, oth] = TOMDEC(sig, freq)
%     [sine, err, har, oth] = TOMDEC(sig, freq, order)
%     [sine, err, har, oth] = TOMDEC(sig, freq, order, disp)
%   or using parameter pairs:
%     [sine, err, har, oth] = TOMDEC(sig, 'name', value, ...)
%
%   Inputs:
%     sig - Signal to be decomposed
%       Vector 
%     freq - Signal frequency (normalized). Optional.
%       Scalar
%       Range: [0, 0.5]
%       Default: auto-detect
%     order - Order of harmonics to fit. Optional.
%       Integer Scalar
%       Default: 10
%     disp - Display switch. Optional.
%       Logical
%       Default: nargout == 0 (auto-display when no outputs)
%
%   Outputs:
%     sine - Fundamental sinewave component (including DC)
%       Vector
%     err - Total error (sig - sine)
%       Vector
%     har - Harmonic distortions
%       Vector
%     oth - All other errors (sig - all harmonics)
%       Vector
%
%   Examples:
%     % Auto-detect frequency and display results
%     tomdec(sig)
%
%     % Specify frequency, fit 10 harmonics
%     [sine, err, har, oth] = tomdec(sig, 0.123)
%
%     % Fit only 5 harmonics without display
%     [sine, err, har, oth] = tomdec(sig, 0.123, 5, false)
%
%   Notes:
%     - The decomposition satisfies: sig = sine + err, err = har + oth
%     - sine contains DC offset and fundamental frequency only
%     - har contains harmonics 2 through order (dependent errors)
%     - oth contains all remaining errors (independent errors)
%     - If freq is not set, the function automatically detects frequency using findfreq
%
%   See also: findfreq, sinefit

    % Parse input arguments
    p = inputParser;
    addOptional(p, 'freq', -1, @(x) isnumeric(x) && isscalar(x) && (x >= 0) && (x <= 0.5));
    addOptional(p, 'order', 10, @(x) isnumeric(x) && isscalar(x) && (x > 0));
    addOptional(p, 'disp', nargout == 0, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    parse(p, varargin{:});
    freq = p.Results.freq;
    order = round(p.Results.order);
    disp = p.Results.disp;

    % Ensure sig is a column vector
    S = size(sig);
    if(S(1) < S(2))
        sig = sig';
    end

    % Auto-detect frequency if not provided
    if(freq < 0)
       freq = findfreq(sig);
    end

    % Time vector
    t = 0:(length(sig)-1);


    % Fit all harmonics up to specified order
    SI = zeros([length(sig),order]);
    SQ = zeros([length(sig),order]);
    for ii = 1:order
      SI(:,ii) = cos(t*freq*ii*2*pi);  % Cosine basis for harmonic ii
      SQ(:,ii) = sin(t*freq*ii*2*pi);  % Sine basis for harmonic ii
    end

    % Solve for weights of all harmonics
    W = linsolve([SI,SQ],sig);

    % DC offset
    DC = mean(sig);

    % Reconstructed fundamental (including DC)
    sine = DC + SI(:,1)*W(1) + SQ(:,1)*W(1+order);

    % Reconstructed signal with all harmonics
    signal_all = DC + [SI,SQ] * W;

    % Compute error components
    err = sig - sine;          % Total error (all non-fundamental)

    har = signal_all - sine;   % Harmonic distortion (2nd through nth harmonics)

    oth = sig - signal_all;    % Other errors (not captured by harmonics)
    
    % Display results if requested
    if(disp)
        % Left y-axis: signal and fitted sine
        yyaxis left;
        plot(sig,'kx');
        hold on;
        plot(sine,'-','color',[0.5,0.5,0.5]);
        axis([1,min(max(1.5/freq,100),length(sig)), min(sig)*1.1, max(sig)*1.1]);
        ylabel('Signal');

        % Right y-axis: error components
        yyaxis right;
        plot(har,'r-');
        hold on;
        plot(oth,'b-');
        axis([1,min(max(1.5/freq,100),length(sig)), min(err)*1.1, max(err)*1.1]);
        ylabel('Error');
        xlabel('Samples');
        legend('signal','sinewave','harmonics','other errors');

    end

end