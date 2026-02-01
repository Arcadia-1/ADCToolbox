function [rep] = adcpanel(dat, varargin)
%ADCPANEL Comprehensive ADC analysis dashboard
%   Unified panel displaying multiple ADC analysis results in a single
%   dashboard. Automatically detects input data type and runs appropriate
%   analysis pipelines.
%
%   Syntax:
%     rep = ADCPANEL(dat)
%     rep = ADCPANEL(dat, 'Name', Value)
%
%   Inputs:
%     dat - ADC data to analyze
%       Vector (N×1 or 1×N): Value-waveform data (time series of samples)
%       Matrix (N×M, N>M): Bit-wise data (N samples, M bits per sample)
%
%   Name-Value Arguments:
%     'dataType' - How to interpret the input data
%       'auto' (default): Auto-detect from dimensions
%       'values': Treat as value-waveform (time series)
%       'bits': Treat as bit-wise data (binary matrix)
%
%     'signalType' - Type of input signal
%       'sinewave' (default): Full sinewave analysis pipeline
%       'other': Basic spectrum analysis only
%
%     'OSR' - Oversampling ratio
%       Positive scalar (default: 1, no oversampling)
%       Defines signal bandwidth as fs/(2*OSR)
%
%     'fs' - Sampling frequency in Hz (inputParser is case-insensitive, 'Fs' also works)
%       Positive scalar (default: 1)
%
%     'maxCode' - Full scale range (max - min)
%       Positive scalar (default: max-min for values, 2^M for bits)
%
%     'harmonic' - Number of harmonics to analyze
%       Positive integer (default: 5)
%
%     'window' - Window function for spectral analysis
%       'hann' (default), 'rect', or function handle (@hamming, etc.)
%
%     'fin' - Normalized input frequency (fin/fs)
%       Scalar in [0, 0.5] (default: 0, auto-detect)
%
%     'disp' - Enable figure display
%       Logical (default: true)
%
%     'verbose' - Enable verbose output
%       Logical (default: false)
%
%   Outputs:
%     rep - Report structure containing:
%       .dataType    - 'values' or 'bits'
%       .signalType  - 'sinewave' or 'other'
%       .spectrum    - Spectral metrics (ENOB, SNDR, SFDR, SNR, THD, etc.)
%       .decomp      - Thompson decomposition (sine, err, har, oth, freq)
%       .errorPhase  - Error analysis with phase binning (emean, erms, anoi, pnoi)
%       .errorValue  - Error analysis with value binning (emean, erms)
%       .linearity   - INL/DNL results (inl, dnl, code)
%       .osr         - OSR sweep results (osr, sndr, sfdr, enob)
%       .phaseFFT    - Phase analysis in FFT mode
%       .phaseLMS    - Phase analysis in LMS mode (with noise circle)
%       .bits        - Bit-wise analysis (weights, offset, overflow) - if bit data
%       .figures     - Handles to all generated figures and axes
%
%   Analysis Pipelines:
%
%   Pipeline A - Value-Waveform + Sinewave:
%     1. plotspec  - Spectrum analysis (ENOB, SNDR, SFDR, SNR, THD)
%     2. tomdec    - Thompson decomposition for time-domain error waveform
%     3. Time-domain plot - Signal and error waveforms (zoomed to max error region)
%     4. errsin    - Sinewave error analysis (both phase and value modes)
%     5. inlsin    - INL/DNL calculation
%     6. perfosr   - Performance vs OSR sweep
%     7. plotphase - Harmonic phase analysis (both FFT and LMS modes)
%
%   Pipeline B - Value-Waveform + Other Signal:
%     1. Time-domain waveform plot
%     2. plotspec  - Basic spectrum display
%
%   Pipeline C - Bit-wise Data:
%     1. bitchk    - Overflow/underflow detection
%     2. wcalsin   - Weight calibration from sinewave
%     3. plotwgt   - Visualize calibrated weights
%     4. If calibration successful: run Pipeline A on calibrated values
%
%   Panel Layout (3×4 grid for sinewave analysis):
%     - Value-waveform + sinewave: 12 panels
%       * time-domain (1)
%       * plotspec (1)
%       * plotphase: FFT + LMS (2)
%       * errsin: phase + value (4 tiles total)
%       * inlsin: INL + DNL (2 tiles total)
%       * perfosr: (2 tiles total)
%     - Value-waveform + other: 2 panels (1×2)
%       * time-domain waveform (1)
%       * plotspec (1)
%     - Bit-wise data: separate figure for bitchk + plotwgt (2 panels)
%       * Main figure: same 3×4 layout as sinewave (if calibration succeeds)
%
%   Examples:
%     % Basic usage with value-waveform data
%     sig = sin(2*pi*0.123*(0:4095)') + 0.01*randn(4096,1);
%     rep = adcpanel(sig);
%
%     % Bit-wise data analysis (quantize sig to bits)
%     bits = dec2bin(round((sig/2.1+0.5)*2^12), 12) - '0';
%     rep = adcpanel(bits);
%
%     % Oversampled data with specific parameters
%     rep = adcpanel(sig, 'OSR', 4, 'fs', 100e6, 'harmonic', 7);
%
%     % Non-sinewave signal (time-domain + spectrum only)
%     noise_sig = randn(4096, 1);
%     rep = adcpanel(noise_sig, 'signalType', 'other');
%
%     % Access specific results from report
%     fprintf('ENOB: %.2f bits\n', rep.spectrum.enob);
%     fprintf('SNDR: %.2f dB\n', rep.spectrum.sndr);
%     fprintf('Max INL: %.3f LSB\n', max(abs(rep.linearity.inl)));
%
%   Notes:
%     - INL/DNL analysis requires integer codes; non-integer data is rounded
%     - Warning issued when N < maxCode, as INL/DNL may be unreliable
%     - For bit-wise data, a separate figure shows bitchk and plotwgt results
%     - Time-domain display zooms to ~3 sine cycles around maximum error point
%
%   See also: plotspec, tomdec, errsin, inlsin, perfosr, plotphase, bitchk, wcalsin, plotwgt

%--------------------------------------------------------------------------
% IMPLEMENTATION PLAN
%--------------------------------------------------------------------------
%
% Step 1: Input Parsing
%   - Use inputParser with addOptional and addParameter
%   - Handle positional vs name-value argument patterns
%   - Validate parameter ranges and types
%   - Create parameter forwarding map for child functions
%
% Step 2: Data Type Detection
%   - If dataType == 'auto':
%     * Vector (any dimension == 1) → 'values'
%     * Matrix (N×M, N>M) → 'bits'
%     * Matrix (N×M, N<M) → transpose, then 'bits'
%   - Validate data dimensions and values
%
% Step 3: Create Panel Figure
%   - Use tiledlayout for flexible panel arrangement
%   - Layout depends on data type and signal type:
%     * Value + sinewave: 8 panels (4×2)
%     * Value + other: 1 panel
%     * Bits: 2 panels for bit analysis + 8 for value (if successful) = 10 panels (5×2)
%   - Handle figure visibility based on 'disp' parameter
%
% Step 4: Execute Analysis Functions
%   - For value-waveform + sinewave (Pipeline A):
%     a) Call plotspec with OSR, fs, maxCode, harmonic, window
%     b) Call tomdec with freq, order=harmonic to get time-domain decomposition
%     c) Create time-domain plot:
%        - Find index of maximum absolute error: idx_max = argmax(|err|)
%        - Calculate samples per cycle: spc = round(1/freq)
%        - Display window: 5 cycles centered at idx_max
%        - idx_start = max(1, idx_max - 2.5*spc)
%        - idx_end = min(N, idx_max + 2.5*spc)
%        - Plot original signal (left y-axis) and error (right y-axis)
%     d) Call errsin with xaxis='phase', osr, window for phase-binned error analysis
%     e) Call errsin with xaxis='value', osr, window for value-binned error analysis
%     f) Call inlsin (warn if N < maxCode)
%     g) Call perfosr with harmonic
%     h) Call plotphase with mode='FFT' for FFT-based coherent phase
%     i) Call plotphase with mode='LMS' for least-squares phase with noise circle
%
%   - For value-waveform + other signal (Pipeline B):
%     a) Plot time-domain waveform (simple time-series)
%     b) Call plotspec (basic spectrum display)
%
%   - For bit-wise data (Pipeline C):
%     a) Call bitchk to check overflow/underflow
%     b) Call wcalsin with freq, order=harmonic to get weights
%     c) Call plotwgt to display calibrated weights with radix annotations
%     d) If calibration successful (all weights same sign, low residual):
%        - Compute postcal = bits * weights' + offset
%        - Run Pipeline A on postcal signal
%   - Each call wrapped in try-catch for graceful error handling
%
% Step 5: Parameter Forwarding Map
%   +---------------+----------+--------+--------+---------+-----------+---------+--------+---------+
%   | adcpanel      | plotspec | errsin | inlsin | perfosr | plotphase | wcalsin | bitchk | tomdec  |
%   +---------------+----------+--------+--------+---------+-----------+---------+--------+---------+
%   | OSR           | ✓        | ✓      |        |         | ✓         |         |        |         |
%   | fs/Fs         | ✓        |        |        |         | ✓         |         |        |         |
%   | maxCode       | ✓        |        |        |         | ✓         |         |        |         |
%   | harmonic      | ✓        |        |        | ✓       | ✓         | → order |        | → order |
%   | window        | ✓        | ✓      |        |         | ✓         |         |        |         |
%   | fin/f0        |          | ✓      |        |         |           | → freq  |        | → freq  |
%   | disp          | (subplot)| (subp) | (subp) | (subp)  | (subp)    |         |        | false   |
%   +---------------+----------+--------+--------+---------+-----------+---------+--------+---------+
%
%   plotphase mode parameter:
%     - Call twice: once with 'mode'='FFT', once with 'mode'='LMS'
%
%   errsin xaxis parameter:
%     - Call twice: once with 'xaxis'='phase', once with 'xaxis'='value'
%
% Step 6: Collect Results into Report Structure
%   - rep.dataType, rep.signalType: input classification
%   - rep.spectrum: outputs from plotspec (enob, sndr, sfdr, snr, thd, sigpwr, noi, nsd)
%   - rep.decomp: outputs from tomdec (sine, err, har, oth, freq)
%   - rep.errorPhase: outputs from errsin with xaxis='phase' (emean, erms, xx, anoi, pnoi)
%   - rep.errorValue: outputs from errsin with xaxis='value' (emean, erms, xx)
%   - rep.linearity: outputs from inlsin (inl, dnl, code)
%   - rep.osr: outputs from perfosr (osr, sndr, sfdr, enob)
%   - rep.phaseFFT: outputs from plotphase with mode='FFT'
%   - rep.phaseLMS: outputs from plotphase with mode='LMS'
%   - rep.bits: outputs from bitchk, wcalsin, plotwgt (if bit-wise)
%     * .overflow: range_min, range_max, ovf_percent_zero, ovf_percent_one
%     * .weights: calibrated weights from wcalsin
%     * .offset: DC offset from wcalsin
%     * .freqcal: refined frequency from wcalsin
%   - rep.figures: handles to panel and individual subplots
%     * .panel: main tiledlayout figure handle
%     * .ax_plotspec, .ax_timedomain, .ax_errsinPhase, .ax_errsinValue
%     * .ax_inlsin, .ax_perfosr, .ax_plotphaseFFT, .ax_plotphaseLMS
%     * .ax_bitchk, .ax_plotwgt (if bit-wise)
%
% Step 7: Finalize
%   - Adjust figure layout if some panels are empty/skipped
%   - Add overall title and parameter summary
%   - Return report struct
%
%--------------------------------------------------------------------------
% EDGE CASES TO HANDLE
%--------------------------------------------------------------------------
%
% 1. Empty or insufficient data
%    → Error with meaningful message
%
% 2. Non-integer data for inlsin
%    → Round with warning, or skip if signalType is 'other'
%
% 3. Rank-deficient bit matrix in wcalsin
%    → Use nomWeight fallback, report in rep.bits.warnings
%
% 4. Non-sinewave data
%    → Only run plotspec, skip errsin/inlsin/perfosr/plotphase
%
% 5. Few data points (N < maxCode)
%    → Warn that INL/DNL results may be unreliable, but still run
%
% 6. Matrix input for value-waveform
%    → Use first column as representative for single-vector functions
%    → Pass full matrix to plotspec for coherent/power averaging
%
%--------------------------------------------------------------------------

    %% Step 1: Input Parsing
    p = inputParser;
    p.KeepUnmatched = false;

    % Data interpretation
    addParameter(p, 'dataType', 'auto', @(x) ischar(x) || isstring(x));
    addParameter(p, 'signalType', 'sinewave', @(x) ischar(x) || isstring(x));

    % Analysis parameters (note: inputParser is case-insensitive, so fs/Fs both work)
    addParameter(p, 'OSR', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'fs', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'maxCode', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
    addParameter(p, 'harmonic', 5, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'window', 'hann', @(x) ischar(x) || isstring(x) || isa(x, 'function_handle'));
    addParameter(p, 'fin', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 0.5);

    % Display options
    addParameter(p, 'disp', true, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'verbose', false, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));

    parse(p, varargin{:});

    % Extract parsed parameters
    dataType = lower(string(p.Results.dataType));
    signalType = lower(string(p.Results.signalType));
    OSR = p.Results.OSR;
    fs = p.Results.fs;
    maxCode = p.Results.maxCode;
    harmonic = round(p.Results.harmonic);
    winType = p.Results.window;
    fin = p.Results.fin;
    dispFlag = logical(p.Results.disp);
    verbose = logical(p.Results.verbose);

    %% Step 2: Data Type Detection and Validation
    if isempty(dat)
        error('adcpanel:emptyData', 'Input data is empty.');
    end

    [N, M] = size(dat);

    % Auto-detect data type
    if dataType == "auto"
        if N == 1 || M == 1
            dataType = "values";
            if N == 1
                dat = dat';
                [N, M] = size(dat);
            end
        else
            if N < M
                dat = dat';
                [N, M] = size(dat);
            end
            dataType = "bits";
        end
    elseif dataType == "bits"
        if N < M
            dat = dat';
            [N, M] = size(dat);
        end
    elseif dataType == "values"
        % Ensure column vector for single signal
        if N == 1 && M > 1
            dat = dat';
            [N, M] = size(dat);
        end
    end

    % For value data, get representative signal (first column if matrix)
    if dataType == "values"
        sig = dat(:, 1);
        sigFull = dat;  % Keep full matrix for plotspec averaging
    end
   

    if verbose
        fprintf('adcpanel: dataType=%s, signalType=%s, N=%d, M=%d\n', dataType, signalType, N, M);
    end

    %% Step 3: Initialize Report Structure
    rep = struct();
    rep.dataType = char(dataType);
    rep.signalType = char(signalType);
    rep.figures = struct();

    %% Step 4: Create Panel Figure
    if dispFlag

        if dataType == "bits"
            % Main 3x4 panel layout for value analysis (same as value-waveform)
            fig = figure('Name', 'ADC Analysis Panel', 'NumberTitle', 'off');
            fig.Position = [50, 50, 1600, 800];
            tl = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact','TileIndexing','columnmajor');

            % For bit-wise: separate figure for bitchk/plotwgt, main 3x4 panel for value analysis
            fig_bits = figure('Name', 'Bit-wise Analysis (bitchk & plotwgt)', 'NumberTitle', 'off');
            fig_bits.Position = [100, 100, 1200, 400];
            tl_bits = tiledlayout(fig_bits, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
            rep.figures.panel_bits = fig_bits;
            rep.figures.tl_bits = tl_bits;
        elseif signalType == "sinewave"
            % 12-panel layout (3x4) for sinewave analysis
            fig = figure('Name', 'ADC Analysis Panel', 'NumberTitle', 'off');
            fig.Position = [50, 50, 1600, 800];
            tl = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact','TileIndexing','columnmajor');
        else
            % 2-panel layout (1x2) for non-sinewave: time-domain + spectrum
            fig = figure('Name', 'ADC Analysis Panel', 'NumberTitle', 'off');
            fig.Position = [50, 50, 1600, 800];
            tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact','TileIndexing','columnmajor');
        end

        rep.figures.panel = fig;
    end

    %% Step 5: Execute Analysis Pipelines

    if dataType == "bits"
        %% Pipeline C: Bit-wise Data Analysis
        bits = dat;
        rep.bits = struct();

        % C1: bitchk - Overflow detection
        try
            ax_bitchk = nexttile(tl_bits);
            rep.figures.ax_bitchk = ax_bitchk;
            [range_min, range_max, ovf_pct_zero, ovf_pct_one] = bitchk(bits, 'disp', true);
            title('Bit Overflow Check');
            rep.bits.overflow = struct('range_min', range_min, 'range_max', range_max, ...
                'ovf_percent_zero', ovf_pct_zero, 'ovf_percent_one', ovf_pct_one);
        catch ME
            warning on;
            warning('adcpanel:bitchkFailed', 'bitchk failed: %s', ME.message);
            rep.bits.overflow = struct('error', ME.message);
        end

        % C2: wcalsin - Weight calibration
        ax_plotwgt = nexttile(tl_bits);
        rep.figures.ax_plotwgt = ax_plotwgt;
        try
            [weights, offset, postcal, ideal, err_wcal, freqcal] = wcalsin(bits, ...
                'freq', fin, 'order', harmonic, 'verbose', verbose);

            if isempty(maxCode)
                % Calculate maxCode and wscalling from weights:
                %
                % Algorithm:
                % 1. Sort absolute weights descending to identify bit significance
                % 2. Find "significant" bits by detecting ratio jumps >= 3
                %    (large jumps indicate transition to noise/redundant bits)
                % 3. Initial wscalling normalizes the smallest significant weight to 1
                % 4. Refine wscalling to minimize rounding error across all weights
                % 5. Compute maxCode as the full-scale range of significant bits
                %
                % This approach handles non-binary and redundant bit architectures
                % by automatically detecting which bits contribute to the signal.

                % Step 1: Sort absolute weights from large to small
                absW = sort(abs(weights), 'descend');

                % Step 2: Calculate ratios between adjacent abs(weights)
                ratios = absW(1:end-1) ./ absW(2:end);

                % Step 3: Find max K such that ratios(i) < 3 for all i = 1 to K
                % K+1 is the number of "significant" bits
                idx = find(ratios >= 3, 1, 'first');
                if isempty(idx)
                    K = length(ratios);  % all ratios < 3, all bits significant
                else
                    K = idx - 1;
                end

                % Step 4: Initial scaling - normalize smallest significant weight to 1
                wscalling = 1 / absW(K+1);

                % Step 5: Refine wscalling to minimize rounding error
                % Search around the initial estimate (0.5x to 1.5x of MSB weight)
                % to find scaling that makes weights closest to integers
                absW_sig = absW(1:K+1);  % only use significant weights for refinement
                w_err = rms(absW_sig * wscalling - round(absW_sig * wscalling));
                WMSB_init = round(absW(1) * wscalling);
                WMSB_min = max(1, round(WMSB_init * 0.5));  % ensure >= 1 to avoid zero scaling
                WMSB_max = max(WMSB_min, round(WMSB_init * 1.5));
                for WMSB = WMSB_min:WMSB_max
                    w_refine = WMSB / absW(1);
                    w_err_ref = rms(absW_sig * w_refine - round(absW_sig * w_refine));
                    if w_err > w_err_ref
                        w_err = w_err_ref;
                        wscalling = w_refine;
                    end
                end

                % Step 6: Calculate maxCode as full-scale range of significant bits
                % maxCode = sum of significant weights after scaling
                maxCode = sum(absW_sig) * wscalling;

            else
                wscalling = maxCode / sum(weights);
            end

            weights = weights * wscalling;
            offset = offset * wscalling;
            rep.bits.weights = weights;
            rep.bits.offset = offset;
            rep.bits.freqcal = freqcal;

            sig = bits*weights';
            sigFull = sig;
            rep.bits.postcal = sig;

            % Use calibrated frequency
            fin = freqcal;

            % C3: plotwgt - Display weights
            plotwgt(weights);
            title('Calibrated Bit Weights');
        catch ME
            warning on;
            warning('adcpanel:wcalsinFailed', 'wcalsin failed: %s', ME.message);
            rep.bits.weights = [];
            rep.bits.error = ME.message;
        end

        % Set signal type to sinewave for calibrated data
        signalType = "sinewave";
    end

    %% Pipeline A/B: Value-waveform Analysis
    if dataType == "values" || dataType == "bits"

        if isempty(maxCode)
            maxCode = max(dat(:)) - min(dat(:));
        end

        if signalType == "sinewave"
            
            %% Pipeline A: Full Sinewave Analysis

            % A2: tomdec - Thompson decomposition (no display, just get data)
            try
                [sine_td, err_td, har_td, oth_td, freq_td] = tomdec(sig, fin, harmonic, false);
                rep.decomp = struct('sine', sine_td, 'err', err_td, 'har', har_td, ...
                    'oth', oth_td, 'freq', freq_td);
                % Update fin if it was auto-detected
                if fin == 0
                    fin = freq_td;
                end
            catch ME
                warning('adcpanel:tomdecFailed', 'tomdec failed: %s', ME.message);
                rep.decomp = struct('error', ME.message);
                err_td = [];
                freq_td = fin;
            end

            % A2b: Time-domain plot (3 cycles around max error)
            if dispFlag
                ax_timedomain = nexttile(tl);
                rep.figures.ax_timedomain = ax_timedomain;

                if ~isempty(err_td) && freq_td > 0
                    % Find max error location
                    [~, idx_max] = max(abs(err_td));

                    % Calculate samples per cycle
                    if freq_td > 0.25
                        spc = round(1/(0.5-freq_td));
                    else
                        spc = round(1 / freq_td);
                    end

                    % Plot ALL data on left axis (signal and ideal)
                    yyaxis left;
                    plot(1:N, sig, 'b.', 'MarkerSize', 4);
                    hold on;
                    plot(1:N, sine_td, 'k-', 'LineWidth', 0.5);
                    ylabel('Signal');
                    % Extend ylim downward to push signal to upper half
                    sig_min = min(sig); sig_max = max(sig);
                    sig_range = sig_max - sig_min;
                    ylim([sig_min - sig_range * 1.1, sig_max + sig_range * 0.1]);
                    hold off;

                    % Plot ALL error data on right axis
                    yyaxis right;
                    plot(1:N, err_td, 'r-', 'LineWidth', 1);
                    ylabel('Error');
                    % Extend ylim upward to push error to lower half
                    err_min = min(err_td); err_max = max(err_td);
                    err_range = err_max - err_min;
                    ylim([err_min - err_range * 0.1, err_max + err_range * 1.1]);

                    % Set x-axis to show 3 cycles centered at max error (user can pan)
                    idx_start = max(1, round(idx_max - max(1.5 * spc, 50)));
                    idx_end = min(N, round(idx_max + max(1.5 * spc, 50)));
                    xlim([idx_start, idx_end]);

                    xlabel('Sample');
                    title(sprintf('Time Domain (@ max err, idx=%d)', idx_max));
                    legend('Signal', 'Ideal', 'Error', 'Location', 'northeast');
                else
                    text(0.5, 0.5, 'Time-domain N/A', 'HorizontalAlignment', 'center');
                    axis off;
                end
            end

            % A3a: errsin (phase mode) - uses nexttile internally
            try
                [emean_p, erms_p, xx_p, anoi, pnoi] = errsin(sig, 'fin', fin, 'xaxis', 'phase', 'osr', OSR, 'window', winType, 'disp', dispFlag);
                rep.errorPhase = struct('emean', emean_p, 'erms', erms_p, 'xx', xx_p, ...
                    'anoi', anoi, 'pnoi', pnoi);
            catch ME
                warning on;
                warning('adcpanel:errsinPhaseFailed', 'errsin(phase) failed: %s', ME.message);
                rep.errorPhase = struct('error', ME.message);
            end

            % A1: plotspec - Spectrum analysis (no subplot, use axes())
            if dispFlag
                ax_plotspec = nexttile(tl);
                rep.figures.ax_plotspec = ax_plotspec;
                axes(ax_plotspec);
            end
            try
                [enob, sndr, sfdr, snr, thd, sigpwr, noi, nsd] = plotspec(sigFull, fs, maxCode, harmonic, ...
                    'OSR', OSR, 'window', winType, 'disp', dispFlag);
                rep.spectrum = struct('enob', enob, 'sndr', sndr, 'sfdr', sfdr, ...
                    'snr', snr, 'thd', thd, 'sigpwr', sigpwr, 'noi', noi, 'nsd', nsd);
                if dispFlag
                    title('Spectrum');
                end
            catch ME
                warning on;
                warning('adcpanel:plotspecFailed', 'plotspec failed: %s', ME.message);
                rep.spectrum = struct('error', ME.message);
            end

            % A3b: errsin (value mode) - uses nexttile internally
            try
                [emean_v, erms_v, xx_v] = errsin(sig, 'fin', fin, 'xaxis', 'value', 'osr', OSR, 'window', winType, 'disp', dispFlag);
                rep.errorValue = struct('emean', emean_v, 'erms', erms_v, 'xx', xx_v);
            catch ME
                warning on;
                warning('adcpanel:errsinValueFailed', 'errsin(value) failed: %s', ME.message);
                rep.errorValue = struct('error', ME.message);
            end

            % A6a: plotphase (FFT mode) - no subplot, use axes()
            if dispFlag
                ax_plotphaseFFT = nexttile(tl);
                rep.figures.ax_plotphaseFFT = ax_plotphaseFFT;
                axes(ax_plotphaseFFT);
            end
            try
                h_fft = plotphase(sigFull, harmonic, maxCode, 'Fs', fs, 'OSR', OSR, 'window', winType, 'mode', 'FFT');
                rep.phaseFFT = struct('handle', h_fft);
                if dispFlag
                    title('Phase Spectrum (FFT mode)');
                end
            catch ME
                warning on;
                warning('adcpanel:plotphaseFFTFailed', 'plotphase(FFT) failed: %s', ME.message);
                rep.phaseFFT = struct('error', ME.message);
            end

            % A4: inlsin - INL/DNL - uses nexttile internally
            if N < maxCode
                warning('adcpanel:fewSamples', ...
                    'N=%d is less than maxCode=%d. INL/DNL results may be unreliable.', N, round(maxCode));
            end
            try
                % Round to integer codes for histogram method
                sig_int = round(sig);
                [inl, dnl, code] = inlsin(sig_int, 0.01, dispFlag);
                rep.linearity = struct('inl', inl, 'dnl', dnl, 'code', code, ...
                    'inl_pp', max(inl) - min(inl), 'dnl_pp', max(dnl) - min(dnl));
            catch ME
                warning on;
                warning('adcpanel:inlsinFailed', 'inlsin failed: %s', ME.message);
                rep.linearity = struct('error', ME.message);
            end

            % A6b: plotphase (LMS mode) - no subplot, use axes()
            if dispFlag
                ax_plotphaseLMS = nexttile(tl);
                rep.figures.ax_plotphaseLMS = ax_plotphaseLMS;
                axes(ax_plotphaseLMS);
            end
            try
                h_lms = plotphase(sigFull, harmonic, maxCode, 'Fs', fs, 'OSR', OSR, 'window', winType, 'mode', 'LMS');
                rep.phaseLMS = struct('handle', h_lms);
                if dispFlag
                    title('Phase Spectrum (LMS mode)');
                end
            catch ME
                warning on;
                warning('adcpanel:plotphaseLMSFailed', 'plotphase(LMS) failed: %s', ME.message);
                rep.phaseLMS = struct('error', ME.message);
            end

            % A5: perfosr - Performance vs OSR - uses nexttile internally
            try
                [osr_vals, sndr_osr, sfdr_osr, enob_osr] = perfosr(sig, ...
                    'harmonic', harmonic, 'disp', dispFlag);
                rep.osr = struct('osr', osr_vals, 'sndr', sndr_osr, ...
                    'sfdr', sfdr_osr, 'enob', enob_osr);
            catch ME
                warning on;
                warning('adcpanel:perfosrFailed', 'perfosr failed: %s', ME.message);
                rep.osr = struct('error', ME.message);
            end

        else
            %% Pipeline B: Non-sinewave (time-domain + spectrum)

            % B1: Time-domain waveform plot
            if dispFlag
                ax_timedomain = nexttile(tl);
                rep.figures.ax_timedomain = ax_timedomain;

                % Simple time-domain plot
                plot(1:N, sig, 'b-', 'LineWidth', 0.5);
                xlabel('Sample');
                ylabel('Signal Value');
                title('Time-Domain Waveform');
                grid on;
            end

            % B2: Spectrum analysis
            if dispFlag
                ax_plotspec = nexttile(tl);
                rep.figures.ax_plotspec = ax_plotspec;
                axes(ax_plotspec);
            end
            try
                [enob, sndr, sfdr, snr, thd, sigpwr, noi, nsd] = plotspec(sigFull, fs, maxCode, 0, ...
                    'OSR', OSR, 'window', winType, 'disp', 1, 'label',0);
                rep.spectrum = [];
                if dispFlag
                    title('Spectrum');
                end
            catch ME
                warning on;
                warning('adcpanel:plotspecFailed', 'plotspec failed: %s', ME.message);
                rep.spectrum = struct('error', ME.message);
            end

            % Mark other fields as not applicable
            rep.decomp = struct('skipped', true, 'reason', 'signalType is not sinewave');
            rep.errorPhase = struct('skipped', true, 'reason', 'signalType is not sinewave');
            rep.errorValue = struct('skipped', true, 'reason', 'signalType is not sinewave');
            rep.linearity = struct('skipped', true, 'reason', 'signalType is not sinewave');
            rep.osr = struct('skipped', true, 'reason', 'signalType is not sinewave');
            rep.phaseFFT = struct('skipped', true, 'reason', 'signalType is not sinewave');
            rep.phaseLMS = struct('skipped', true, 'reason', 'signalType is not sinewave');
        end
    end

    %% Step 6: Finalize
    if dispFlag
        % Add overall title
        if dataType == "bits"
            title(tl, sprintf('ADC Panel: Bit-wise Data (%d samples, %d bits)', N, M));
        else
            title(tl, sprintf('ADC Panel: %s Signal (%d samples)', ...
                upper(char(signalType)), N));
        end
    end

end