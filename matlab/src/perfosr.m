function [osr, sndr, sfdr, enob] = perfosr(sig, varargin)
%PERFOSR Plot ADC performance metrics versus oversampling ratio (OSR)
%   This function analyzes how ADC performance (SNDR, SFDR, ENOB) varies
%   with oversampling ratio. It separates the ideal signal from error using
%   sine fitting, then calculates metrics at different bandwidths.
%
%   Syntax:
%     [osr, sndr, sfdr, enob] = PERFOSR(sig)
%     [osr, sndr, sfdr, enob] = PERFOSR(sig, 'Name', Value)
%
%   Inputs:
%     sig - Signal to be analyzed (typically ADC output data)
%       Vector of real numbers
%
%   Name-Value Arguments:
%     'disp' - Enable plotting
%       Logical or numeric (0 or 1)
%       Default: true when nargout == 0
%     'osr' - List of OSR values to sweep
%       Vector of positive numbers
%       Default: N/2 ./ (N/2:-1:1), where N is length of sig
%     'logscale' - Use logarithmic scale on x-axis
%       Logical or numeric (0 or 1)
%       Default: true
%     'harmonic' - Number of harmonics to mark on plot
%       Positive integer
%       Default: 5
%     'smooth' - Smoothing window size for slope calculation (number of points)
%       Positive integer
%       Default: round(length(osr)/20), minimum 5
%
%   Outputs:
%     osr - Oversampling ratio values
%       Vector (1 x M)
%     sndr - SNDR at each OSR in dB
%       Vector (1 x M)
%     sfdr - SFDR at each OSR in dB
%       Vector (1 x M)
%     enob - ENOB at each OSR
%       Vector (1 x M)
%
%   Examples:
%     % Basic usage with auto-display
%     perfosr(sig);
%
%     % Custom OSR values
%     [osr, sndr, sfdr, enob] = perfosr(sig, 'osr', [2 4 8 16 32 64]);
%
%     % Linear x-axis scale
%     perfosr(sig, 'logscale', false);
%
%   Notes:
%     - Signal power is determined by ideal sine fit (constant across OSR)
%     - Noise/spur power only includes bins within the current OSR bandwidth
%     - SNDR slope is estimated and zones are indicated by different slopes
%     - Signal frequency and harmonics are marked on the plot
%
%   See also: plotspec, sinfit, alias

    % Input validation
    if ~isnumeric(sig) || ~isreal(sig)
        error('perfosr:invalidInput', 'Input signal must be a real numeric array.');
    end

    % Reshape to column vector if needed
    [N, M] = size(sig);
    if M > 1 && N == 1
        sig = sig';
        N = M;
    end

    % Input parsing
    p = inputParser;
    addParameter(p, 'disp', nargout == 0, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'osr', [], @(x) isnumeric(x) && all(x > 0));
    addParameter(p, 'logscale', true, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'harmonic', 5, @(x) isnumeric(x) && isscalar(x) && x > 0 && mod(x,1) == 0);
    addParameter(p, 'smooth', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
    parse(p, varargin{:});

    dispPlot = logical(p.Results.disp);
    logscale = logical(p.Results.logscale);
    harmonic = p.Results.harmonic;
    smoothWin = p.Results.smooth;

    % Default OSR sweep: from 1 to N/2
    if isempty(p.Results.osr)
        n_bins = floor(N/2);
        osr = (N/2) ./ (n_bins:-1:1);
    else
        osr = p.Results.osr;
    end

    % Step 1: Fit sine wave to get ideal signal and frequency
    [sig_fit, fin, mag] = sinfit(sig);

    % Step 2: Calculate error (residual)
    err = sig - sig_fit;

    % Step 3: Apply Hann window and compute FFT of error
    win = hannwin(N);
    err_windowed = err .* win' / sqrt(mean(win.^2));
    err_spec = abs(fft(err_windowed)).^2 / N^2 * 4;  % Power spectrum (one-sided scaling)
    err_spec = err_spec(1:floor(N/2)+1);  % Keep positive frequencies

    % Signal power (constant, determined by ideal sine fit)
    sig_power = mag^2 / 2;  % RMS power of sine wave

    % Step 4-5: Sweep OSR and calculate metrics (optimized: high OSR to low OSR)
    n_osr = length(osr);
    sndr = zeros(1, n_osr);
    sfdr = zeros(1, n_osr);
    enob = zeros(1, n_osr);

    % Sort OSR descending (high to low) for incremental calculation
    [osr_sorted, sort_idx] = sort(osr, 'descend');

    % Initialize with highest OSR (smallest bandwidth)
    n_inband_prev = 0;
    noi_power = 0;
    spur_power = 0;

    for ii = 1:n_osr
        % Number of bins in band for this OSR
        n_inband = floor(N / 2 / osr_sorted(ii));
        n_inband = max(1, min(n_inband, length(err_spec)));

        % Incremental calculation: only process new bins since last OSR
        if n_inband > n_inband_prev
            incremental_spec = err_spec(n_inband_prev+1:n_inband);
            noi_power = noi_power + sum(incremental_spec);
            spur_power = max(spur_power, max(incremental_spec));
        end
        n_inband_prev = n_inband;

        % Calculate metrics and store in sorted order
        orig_idx = sort_idx(ii);
        sndr(orig_idx) = 10 * log10(sig_power / noi_power);
        sfdr(orig_idx) = 10 * log10(sig_power / spur_power);
        enob(orig_idx) = (sndr(orig_idx) - 1.76) / 6.02;
    end

    % Step 6-8: Plot results
    if dispPlot

        % Subplot 1: Performance metrics (SNDR, SFDR, ENOB)
        nexttile;

        % Plot SNDR and SFDR on left axis
        if logscale
            yyaxis left;
            semilogx(osr, sndr, 'b-', 'LineWidth', 1.5);
            hold on;
            semilogx(osr, sfdr, 'r-', 'LineWidth', 1.5);
        else
            yyaxis left;
            plot(osr, sndr, 'b-', 'LineWidth', 1.5);
            hold on;
            plot(osr, sfdr, 'r-', 'LineWidth', 1.5);
        end
        ylabel('SNDR / SFDR (dB)');

        % Set up right y-axis to show ENOB (proportional to SNDR)
        % ENOB = (SNDR - 1.76) / 6.02
        yyaxis right;
        sndr_lim = [min([sndr, sfdr]) - 5, max([sndr, sfdr]) + 5];
        enob_lim = (sndr_lim - 1.76) / 6.02;
        ylim(enob_lim);
        ylabel('ENOB (bits)');

        % Set left axis limits to match
        yyaxis left;
        ylim(sndr_lim);

        title('Performance vs OSR');
        grid on;
        legend('SNDR (ENOB)', 'SFDR', 'Location', 'southeast');

        % Step 8: Mark signal frequency and harmonics
        % Calculate OSR values corresponding to signal and harmonics
        % OSR where fin is at Nyquist: OSR = Fs/2 / fin = N/2 / (fin*N) = 1/(2*fin)
        osr_sig = 1 / (2 * fin);  % OSR where signal is at Nyquist

        yyaxis left;
        hold on;

        % Mark signal frequency
        if osr_sig >= min(osr) && osr_sig <= max(osr)
            y_lim = ylim;
            if logscale
                semilogx([osr_sig, osr_sig], y_lim, 'k-', 'LineWidth', 1, 'HandleVisibility', 'off');
            else
                plot([osr_sig, osr_sig], y_lim, 'k-', 'LineWidth', 1, 'HandleVisibility', 'off');
            end
            text(osr_sig, y_lim(1), 'Fund', 'Color', 'k', 'FontSize', 8, ...
                'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
        end

        % Mark harmonics (up to Fs/2)
        for h = 2:harmonic
            % Harmonic frequency (may alias)

            osr_h = 1 / (2 * alias(fin * h, 1));

            if osr_h >= min(osr) && osr_h <= max(osr)
                y_lim = ylim;
                if logscale
                    semilogx([osr_h, osr_h], y_lim, 'k--', 'LineWidth', 0.5, 'HandleVisibility', 'off');
                else
                    plot([osr_h, osr_h], y_lim, 'k--', 'LineWidth', 0.5, 'HandleVisibility', 'off');
                end
                text(osr_h, y_lim(1), sprintf('HD%d', h), 'Color', 'k', 'FontSize', 8, ...
                    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
            end
        end

        % Subplot 2: SNDR slope (dB/decade)
        nexttile;

        if length(osr) >= 3
            log_osr = log10(osr);
            n_pts = length(osr);

            % Determine smoothing window size (half-width for central difference)
            if isempty(smoothWin)
                smoothWin = max(5, round(n_pts / 10));
            end
            smoothWin = min(smoothWin, floor((n_pts - 1) / 2));  % Cap to avoid out-of-range

            % Calculate slope using central difference with smoothWin span
            % slope[i] = (sndr[i+smoothWin] - sndr[i-smoothWin]) / (log_osr[i+smoothWin] - log_osr[i-smoothWin])
            local_slope = zeros(1, n_pts);
            for i = 1:n_pts
                i_lo = max(1, i - smoothWin);
                i_hi = min(n_pts, i + smoothWin);
                local_slope(i) = (sndr(i_hi) - sndr(i_lo)) / (log_osr(i_hi) - log_osr(i_lo));
            end

            % Plot slope curve
            if logscale
                semilogx(osr, local_slope, 'b-', 'LineWidth', 1.5);
                hold on;
                % White noise limit reference line at 10 dB/decade
                semilogx([min(osr), max(osr)], [10, 10], 'k--', 'LineWidth', 0.5);
            else
                plot(osr, local_slope, 'b-', 'LineWidth', 1.5);
                hold on;
                plot([min(osr), max(osr)], [10, 10], 'k--', 'LineWidth', 0.5);
            end
            text(max(osr), 10, 'White Noise Limit', 'FontSize', 8, ...
                'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');

            ylabel('SNDR Slope (dB/decade)');
            grid on;

            % Set reasonable y-axis limits
            slope_range = max(local_slope) - min(local_slope);
            ylim([min(local_slope) - 0.1*slope_range - 5, max(local_slope) + 0.1*slope_range + 5]);
        end

        xlabel('OSR');
    end
end

% Embedded Hann window function (no toolbox required)
function w = hannwin(N)
    if N == 1
        w = 1;
    else
        n = 0:(N-1);
        w = 0.5 * (1 - cos(2*pi*n/N));
    end
end
