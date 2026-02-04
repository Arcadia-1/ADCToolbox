function [sine, err, har, oth, freq] = tomdec(sig, varargin)
%TOMDEC Thompson decomposition of a single-tone signal into sinewave and errors
%   This function performs Thompson decomposition on a single-tone signal,
%   separating the signal into fundamental sinewave, harmonic distortions, and other
%   errors. The method uses least-squares fitting to decompose the signal.
%
%   Syntax:
%     [sine, err, har, oth, freq] = TOMDEC(sig)
%     [sine, err, har, oth, freq] = TOMDEC(sig, freq)
%     [sine, err, har, oth, freq] = TOMDEC(sig, freq, order)
%     [sine, err, har, oth, freq] = TOMDEC(sig, freq, order, disp)
%   or using parameter pairs:
%     [sine, err, har, oth, freq] = TOMDEC(sig, 'name', value, ...)
%
%   Inputs:
%     sig - Signal to be decomposed
%       Vector
%     freq - Signal frequency (normalized). Optional.
%       Scalar
%       Range: [0, 0.5]
%       Default: auto-detect
%     order - Order of harmonics to fit. Optional.
%       Integer Scalar (1 means no harmonic separation)
%       Default: 1
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
%     freq - Signal frequency (normalized)
%       Scalar
%
%   Examples:
%     % Auto-detect frequency and display results
%     tomdec(sig)
%
%     % Specify frequency, fit 10 harmonics
%     [sine, err, har, oth, freq] = tomdec(sig, 0.123)
%
%     % Fit only 5 harmonics without display
%     [sine, err, har, oth, freq] = tomdec(sig, 0.123, 5, false)
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
    addOptional(p, 'order', 1, @(x) isnumeric(x) && isscalar(x) && (x >= 1));
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
    if(freq <= 0)
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
    W = linsolve([ones(length(sig),1),SI,SQ],sig);

    % DC offset
    DC = W(1);

    % Reconstructed fundamental (including DC)
    sine = DC + SI(:,1)*W(2) + SQ(:,1)*W(2+order);

    % Reconstructed signal with all harmonics
    signal_all = DC + [SI,SQ] * W(2:end);

    % Compute error components
    err = sig - sine;          % Total error (all non-fundamental)

    har = signal_all - sine;   % Harmonic distortion (2nd through nth harmonics)

    oth = sig - signal_all;    % Other errors (not captured by harmonics)
    
    % Display results if requested
    if(disp)
        N = length(sig);

        % Find max error location
        [~, idx_max] = max(abs(err));

        % Calculate samples per cycle (consider aliasing)
        if freq > 0.25
            spc = round(1 / (0.5 - freq));
        else
            spc = round(1 / freq);
        end

        % Left y-axis: signal and fitted sine
        yyaxis left;
        plot(1:N, sig, 'b.', 'MarkerSize', 4);
        hold on;
        plot(1:N, sine, 'k-', 'LineWidth', 0.5);
        ylabel('Signal');
        % Extend ylim downward to push signal to upper half
        sig_min = min(sig); sig_max = max(sig);
        sig_range = sig_max - sig_min;
        ylim([sig_min - sig_range * 1.1, sig_max + sig_range * 0.1]);
        hold off;

        % Right y-axis: error components
        yyaxis right;
        if order == 1
            % Only plot total error when no harmonic separation
            plot(1:N, err, 'r-', 'LineWidth', 1);
            leg_entries = {'Signal', 'Ideal', 'Error'};
            err_min = min(err); err_max = max(err);
        else
            % Plot har and oth overlapped
            plot(1:N, har, 'r-', 'LineWidth', 1);
            hold on;
            plot(1:N, oth, 'm-', 'LineWidth', 1);
            hold off;
            leg_entries = {'Signal', 'Ideal', 'Harmonics', 'Other'};
            err_min = min(min(har),min(oth)); err_max = max(max(har),max(oth));
        end
        ylabel('Error');
        % Extend ylim upward to push errors to lower half
        err_range = err_max - err_min;
        ylim([err_min - err_range * 0.1, err_max + err_range * 1.1]);

        % Set x-axis to show 3 cycles centered at max error
        idx_start = max(1, round(idx_max - max(1.5 * spc, 50)));
        idx_end = min(N, round(idx_max + max(1.5 * spc, 50)));
        xlim([idx_start, idx_end]);

        xlabel('Sample');
        title(sprintf('Thompson Decomposition (@ max err, idx=%d)', idx_max));
        legend(leg_entries, 'Location', 'northeast');
    end

end