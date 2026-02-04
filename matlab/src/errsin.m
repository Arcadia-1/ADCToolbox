function [emean, erms, xx, anoi, pnoi, err, errxx] = errsin(sig, varargin)
%ERRSIN Analyze sinewave fit errors with histogram binning
%   This function fits a sinewave to the input signal and analyzes the
%   residual errors, binning them by either phase or value. It computes
%   mean and RMS errors per bin, and estimates amplitude and phase noise
%   components when using phase mode.
%
%   Syntax:
%     [emean, erms, xx, anoi, pnoi, err, errxx] = ERRSIN(sig)
%     [emean, erms, xx, anoi, pnoi, err, errxx] = ERRSIN(sig, Name, Value)
%
%   Inputs:
%     sig - Signal to be analyzed (typically ADC output or sinewave samples)
%       Vector of real numbers
%
%   Name-Value Arguments:
%     'bin' - Number of bins for histogram analysis
%       Positive integer (default: 100)
%     'fin' - Normalized input frequency (frequency/fs)
%       Positive scalar in range (0,1). If 0 or omitted, frequency is
%       automatically estimated (default: 0)
%     'disp' - Display plots of error analysis
%       Logical or numeric (0 or 1)
%       Default: nargout == 0 (auto-display when no outputs)
%     'xaxis' - X-axis mode for binning
%       'phase' = bin by phase angle (default)
%       'value' = bin by signal value
%     'erange' - Error range filter for output err and errxx
%       2-element vector [min, max] to filter x-axis values. If empty, no
%       filtering is applied (default: [])
%     'osr' - Oversampling rate
%       Positive scalar >= 1. When osr > 1, the residual error is filtered
%       by an ideal low-pass filter (via ifilter) with passband [0, Fs/2/osr]
%       before analysis. (default: 1, no filtering)
%     'window' - Window function for error signal before filtering
%       String: 'hann' (Hanning window) or 'rect' (Rectangle window)
%       Function handle: e.g., @blackman, @hamming (requires Signal Processing Toolbox)
%       Only applied when osr > 1. Window reduces edge discontinuities before
%       FFT-based filtering, then is undone after filtering to restore noise
%       magnitude. Regularization limits edge amplification. (default: 'hann')
%
%   Outputs:
%     emean - Mean error for each bin
%       Vector (1×bin)
%     erms - RMS error for each bin
%       Vector (1×bin)
%     xx - Bin centers, i.e., the x-axis for emean and erms
%       Vector (1×bin)
%       Phase mode: [0, 360), Value mode: centered bin values
%     anoi - Estimated amplitude noise RMS
%       Scalar. In phase mode, estimated from error pattern. In code mode,
%       returns NaN
%     pnoi - Estimated phase noise RMS (radians)
%       Scalar. In phase mode, normalized by signal magnitude. In code mode,
%       returns NaN
%     err - Raw errors for each sample point (sig_fit - sig)
%       Vector (N×1), filtered by erange if specified
%     errxx - X-axis values corresponding to err
%       Vector (N×1), same size as err
%       Phase mode: phase in degrees, Value mode: sig values
%
%   Examples:
%     % Basic error analysis with phase binning
%     sig = sin(2*pi*0.12345*(0:999)') + 0.01*randn(1000,1);
%     [emean, erms, xx, anoi, pnoi] = errsin(sig);
%
%     % Value mode analysis with 50 bins and turn on display
%     [emean, erms, xx] = errsin(sig, 'xaxis', 'value', 'bin', 50, 'disp', 1);
%
%     % Filter errors to specific phase range
%     [~, ~, ~, ~, ~, err, errxx] = errsin(sig, 'erange', [90, 180]);
%
%     % Analyze oversampled signal with OSR=16 and Hann window
%     [emean, erms, xx] = errsin(sig, 'osr', 16, 'window', 'hann');
%
%   Notes:
%     - Phase mode (xaxis = 'phase'):
%       * Bins errors by phase angle of the sinewave
%       * Estimates amplitude and phase noise using least squares fit:
%         erms^2 = anoi^2*cos^2(theta) + pnoi^2*sin^2(theta)
%     - Value mode (xaxis = 'value'):
%       * Bins errors by the signal value
%       * Useful for analyzing INL
%       * anoi and pnoi are set to NaN as they cannot be estimated
%     - The function automatically orients input signal to column vector
%     - If fin=0, sinfit automatically estimates the frequency
%     - When osr > 1:
%       * Window is applied before filtering to reduce edge discontinuities
%       * Ideal low-pass filter extracts in-band noise (passband [0, Fs/2/osr])
%       * Window is undone after filtering to restore noise magnitude scale
%       * Regularization (max(win, 0.01)) prevents excessive edge amplification
%
%   See also: sinfit, inlsin, ifilter

    % Input validation
    if ~isreal(sig)
        error('errsin:invalidInput', 'Signal must be real.');
    end

    % Input parsing
    p = inputParser;
    addOptional(p, 'bin', 100, @(x) isnumeric(x) && isscalar(x) && (x > 0));
    addOptional(p, 'fin', 0, @(x) isnumeric(x) && isscalar(x) && (x > 0) && (x < 1));
    addOptional(p, 'disp', nargout == 0, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'xaxis', 'phase', @(x) ischar(x) && (strcmpi(x,'phase') || strcmpi(x,'value')));
    addParameter(p, 'erange', []);  % err filter, only the errors in erange are returned to err
    addParameter(p, 'osr', 1, @(x) isnumeric(x) && isscalar(x) && (x >= 1));
    addParameter(p, 'window', 'hann', @(x) (ischar(x) && ismember(x, {'hann', 'rect'})) || isa(x, 'function_handle'));
    parse(p, varargin{:});
    bin = round(p.Results.bin);
    fin = p.Results.fin;
    disp = p.Results.disp;
    xaxis = lower(p.Results.xaxis);
    erange = p.Results.erange;
    osr = p.Results.osr;
    windowFunc = p.Results.window;

    % Ensure column vector orientation
    S = size(sig);
    if(S(1) < S(2))
        sig = sig';
    end

    % Fit sine wave to input signal
    if(fin == 0)
        [sig_fit,fin,mag,~,phi] = sinfit(sig);
    else
        [sig_fit,~,mag,~,phi] = sinfit(sig,fin);
    end

    % Calculate residual errors
    err = sig_fit-sig;

    % Apply low-pass filter if oversampling rate > 1
    if osr > 1
        % Generate window function
        N = length(err);
        if ischar(windowFunc)
            % Use embedded window functions
            if strcmp(windowFunc, 'hann')
                win = hannwin(N);
            elseif strcmp(windowFunc, 'rect')
                win = ones(1, N);
            else
                win = ones(1, N);
                warning('errsin:invalidWindow', 'Unknown window type ''%s'', using rectangular window', windowFunc);
            end
        else
            % Use function handle (requires Signal Processing Toolbox)
            try
                win = window(windowFunc, N, 'periodic')';
            catch
                try
                    win = window(windowFunc, N)';
                catch
                    win = ones(1, N);
                    warning('errsin:windowFailed', 'Failed to generate window, using rectangular window');
                end
            end
        end

        % Apply window, filter, then undo window to restore magnitude
        % Window reduces edge discontinuities before FFT-based filtering
        err = err .* win';
        err = ifilter(err, [0, 0.5/osr]);
        % Undo window with regularization to prevent division by very small values
        % Use max(win, 0.01) to limit amplification at window edges
        err = err ./ max(win, 0.01)';
    end

    % Calculate phase for each sample point (used by both modes)
    phase = mod(phi/pi*180 + (0:length(sig)-1)'*fin*360, 360);

    % Value mode: bin by signal value
    if(strcmp(xaxis,'value'))
        errxx = sig;
        dat_min = min(sig);
        dat_max = max(sig);
        bin_wid = (dat_max-dat_min)/bin;
        xx = min(sig) + (1:bin)*bin_wid - bin_wid/2;

        enum = zeros(1,bin);
        esum = zeros(1,bin);
        erms = zeros(1,bin);
        % Separate accumulators for rising [0,180) and falling [180,360) phases
        enum_rise = zeros(1,bin);
        esum_rise = zeros(1,bin);
        enum_fall = zeros(1,bin);
        esum_fall = zeros(1,bin);

        % Accumulate errors for each bin
        for ii = 1:length(sig)
            b = min(floor((sig(ii)-dat_min)/bin_wid)+1,bin);
            esum(b) = esum(b) + err(ii);
            enum(b) = enum(b) + 1;
            % Accumulate separately by phase
            if phase(ii) < 180
                esum_rise(b) = esum_rise(b) + err(ii);
                enum_rise(b) = enum_rise(b) + 1;
            else
                esum_fall(b) = esum_fall(b) + err(ii);
                enum_fall(b) = enum_fall(b) + 1;
            end
        end
        emean = esum./enum;
        emean_rise = esum_rise./enum_rise;
        emean_fall = esum_fall./enum_fall;

        % Calculate RMS error for each bin
        for ii = 1:length(sig)
            b = min(floor((sig(ii)-dat_min)/bin_wid)+1,bin);
            erms(b) = erms(b) + (err(ii) - emean(b))^2;
        end
        erms = sqrt(erms./enum);

        % No noise estimation in code mode
        anoi = nan;
        pnoi = nan;

        % Apply error range filter if specified
        if(~isempty(erange))
            eid = (errxx >= erange(1)) & (errxx <= erange(2));
            errxx = errxx(eid);
            err = err(eid);
        end

        % Display plots if requested
        if(disp)

            nexttile;

            % Plot data points colored by phase (no legend)
            idx_first = (phase < 180);   % [0, 180) phase
            idx_second = (phase >= 180); % [180, 360) phase
            plot(sig(idx_first), err(idx_first), '.', 'Color', [1,0.5,0.5], 'HandleVisibility', 'off');
            hold on;
            plot(sig(idx_second), err(idx_second), '.', 'Color', [0.5,0.5,1], 'HandleVisibility', 'off');

            % Plot averaged traces
            plot(xx, emean_rise, 'r-', 'LineWidth', 1.5);
            plot(xx, emean_fall, 'b-', 'LineWidth', 1.5);
            plot(xx, emean, 'k-', 'LineWidth', 1.5);
            axis([dat_min, dat_max, min(err)*1.05-max(err)*0.05, max(err)*1.05-min(err)*0.05]);
            ylabel('error');
            xlabel('value');

            if(~isempty(erange))
                plot(errxx, err, 'm.', 'HandleVisibility', 'off');
            end

            legend('Rising aveg. (0~180 deg)', 'Falling aveg. (180~360 deg)', 'All aveg.');

            nexttile;
            bar(xx, erms, 'FaceColor', [0.8 0.8 0.8], 'EdgeColor', [0.6 0.6 0.6]);
            axis([dat_min, dat_max, 0, max(erms)*1.1]);
            xlabel('value');
            ylabel('RMS error');


        end

    % Phase mode: bin by phase angle
    else
        errxx = phase';  % Use pre-calculated phase
        xx = (0:bin-1)/bin*360;

        enum = zeros(1,bin);
        esum = zeros(1,bin);
        erms = zeros(1,bin);

        % Accumulate errors for each phase bin
        for ii = 1:length(sig)
            b = mod(round(errxx(ii)/360*bin),bin)+1;
            esum(b) = esum(b) + err(ii);
            enum(b) = enum(b) + 1;
        end
        emean = esum./enum;

        % Calculate RMS error for each phase bin
        for ii = 1:length(sig)
            b = mod(round(errxx(ii)/360*bin),bin)+1;
            erms(b) = erms(b) + (err(ii) - emean(b))^2;
        end
        erms = sqrt(erms./enum);

        % Estimate amplitude and phase noise components
        % Amplitude noise affects all phases equally (cos^2 pattern)
        % Phase noise creates errors proportional to slope (sin^2 pattern)
        asen = cos(xx/360*2*pi).^2;    % amplitude noise sensitivity
        psen = sin(xx/360*2*pi).^2;    % phase noise sensitivity

        % Least squares fit: erms^2 = anoi^2*asen + pnoi^2*psen
        tmp = linsolve([asen', psen'], erms'.^2);

        % Warn if noise power is negative (unusual, may indicate fitting issues)
        if tmp(1) < 0
            warning('errsin:negativeNoise', ...
                'Amplitude noise power is negative (%.2e), clamped to zero.', tmp(1));
        end
        if tmp(2) < 0
            warning('errsin:negativeNoise', ...
                'Phase noise power is negative (%.2e), clamped to zero.', tmp(2));
        end

        anoi = sqrt(max(tmp(1), 0));  % Clamp to non-negative before sqrt
        pnoi = sqrt(max(tmp(2), 0));

        % Apply error range filter if specified
        if(~isempty(erange))
            eid = (errxx >= erange(1)) & (errxx <= erange(2));
            errxx = errxx(eid);
            err = err(eid);
        end

        % Display plots if requested
        if(disp)
            
            nexttile;

            yyaxis left;
            plot(errxx,sig,'k.');
            axis([0,360,min(sig)*1.05-max(sig)*0.05, max(sig)*1.05-min(sig)*0.05]);
            ylabel('data');

            yyaxis right;
            plot(errxx,err,'r.');
            hold on;
            plot(xx,emean,'b-','LineWidth', 1.5);
            axis([0,360,min(err)*1.05-max(err)*0.05, max(err)*1.05-min(err)*0.05]);
            ylabel('error');

            legend('Data','Error','Error aver.');
            xlabel('phase(deg)');

            if(~isempty(erange))
                plot(errxx,err,'m.');
            end

            nexttile;
            bar(xx, erms, 'FaceColor', [0.8 0.8 0.8], 'EdgeColor', [0.6 0.6 0.6]);
            hold on;
            plot(xx, sqrt((anoi.^2)*asen + (pnoi.^2)*psen), 'k-', 'LineWidth', 2);
            plot(xx, sqrt((anoi.^2)*asen), 'b-', 'LineWidth', 1.5);
            plot(xx, sqrt((pnoi.^2)*psen), 'r-', 'LineWidth', 1.5);
            axis([0,360,0,max(erms)*1.5]);
            legend('RMS error', 'Fit', 'Amplitude noise', 'Phase noise', 'Location', 'southeast');
            text(10, max(erms)*1.4, sprintf('Normalized Amplitude Noise RMS = %.2e', anoi/mag), 'color', [0,0,1]);
            text(10, max(erms)*1.25, sprintf('Phase Noise RMS = %.2e rad', pnoi/mag), 'color', [1,0,0]);

            % Display excessive noise if one component dominates
            noiseDiff = anoi^2 - pnoi^2;
            noiseSum = anoi^2 + pnoi^2;
            if abs(noiseDiff) >= 0.05 * abs(noiseSum)
                noiseExcessive = sqrt(abs(noiseDiff));
                if noiseDiff > 0
                    text(10, max(erms)*1.1, sprintf('Excessive Amplitude Noise = %.2e', noiseExcessive/mag), 'color', [0,0,0.5]);
                else
                    text(10, max(erms)*1.1, sprintf('Excessive Phase Noise = %.2e rad', noiseExcessive/mag), 'color', [0.5,0,0]);
                end
            end

            xlabel('phase(deg)');
            ylabel('RMS error');
        end
    end

    % Nested function for embedded Hanning window (no toolbox required)
    function w = hannwin(N)
        % HANNWIN Embedded Hanning window function
        %   w = HANNWIN(N) returns an N-point Hanning (raised cosine) window
        %   in a row vector. This is a simple embedded implementation that
        %   doesn't require the Signal Processing Toolbox
        %
        %   The Hanning window is defined as:
        %   w(n) = 0.5 * (1 - cos(2*pi*n/(N-1))) for n = 0, 1, ..., N-1
        if N == 1
            w = 1;
        else
            n = 0:(N-1);
            w = 0.5 * (1 - cos(2*pi*n/N));
        end
    end

end
