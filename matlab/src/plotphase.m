function [h] = plotphase(sig,varargin)
%PLOTPHASE Plot coherent phase spectrum with polar display
%   This function performs coherent spectral phase analysis on ADC data
%   and displays the result in a polar plot. It aligns the phase of multiple
%   measurement runs and shows harmonics on a polar coordinate system.
%
%   Syntax:
%     h = PLOTPHASE(sig)
%     h = PLOTPHASE(sig, harmonic)
%     h = PLOTPHASE(sig, harmonic, maxSignal)
%     h = PLOTPHASE(sig, 'Name', Value)
%
%   Inputs:
%     sig - Signal to be analyzed, typically the ADC's output data
%       Vector or Matrix (N_run x N_samples)
%       Each row represents a separate measurement run
%       The FFT length is determined by the number of samples
%
%   Optional Positional Inputs:
%     harmonic - Number of harmonics to display. Default: 5
%       Scalar, positive integer
%     maxSignal - Full scale range (max-min). Default: max(sig)-min(sig)
%       Scalar, positive real number
%
%   Name-Value Parameters:
%     'Fs' - Sampling frequency in Hz. Default: 1
%       Scalar, positive real number
%       Used for frequency-based cutoff filtering
%     'mode' - Analysis mode. Default: 'LMS'
%       String: 'FFT' (FFT-based coherent averaging) or 'LMS' (Least squares fitting)
%       'FFT': Traditional FFT with coherent averaging and phase alignment
%       'LMS': Least squares harmonic fitting, displays noise circle on polar plot
%     'OSR' - Oversampling ratio. Default: 1 (no oversampling)
%       Scalar, positive real number
%       Defines signal bandwidth as Fs/(2*OSR)
%     'cutoff' - High-pass cutoff frequency for low-frequency noise removal in Hz. Default: 0
%       Scalar, non-negative real number
%       Removes flicker noise (1/f noise) below the specified frequency
%     'window' - Window function for spectral processing. Default: 'hann'
%       String: 'hann' (Hanning window) or 'rect' (Rectangle window)
%       Function handle: e.g., @blackman, @hamming (requires Signal Processing Toolbox)
%       Window is normalized to preserve RMS power
%       FFT mode: Applied before FFT to reduce spectral leakage
%       LMS mode: Applied before OSR filtering (when OSR > 1) to reduce edge effects
%
%   Outputs:
%     h - Plot handle
%       Graphics handle to the polar plot
%
%   Examples:
%     % Basic usage with default parameters
%     h = plotphase(sig);
%
%     % Specify number of harmonics and full scale
%     h = plotphase(sig, 7, 2^16);
%
%     % Use with oversampling
%     h = plotphase(sig, 'OSR', 32);
%
%     % Remove low-frequency noise below 1 kHz
%     h = plotphase(sig, 'Fs', 100e6, 'cutoff', 1e3);
%
%     % Use least squares fitting mode with noise circle
%     h = plotphase(sig, 7, 'mode', 'LMS');
%
%     % LMS mode with oversampling (filters residual to signal bandwidth)
%     h = plotphase(sig, 'mode', 'LMS', 'OSR', 32);
%
%     % Combine positional and name-value parameters
%     h = plotphase(sig, 10, 2^16, 'Fs', 50e6, 'OSR', 64, 'cutoff', 500);
%
%     % Use rectangular window in FFT mode
%     h = plotphase(sig, 'mode', 'FFT', 'window', 'rect');
%
%   Notes:
%     - Signal can be provided as a row vector, column vector, or matrix
%     - For matrix input, each row is treated as a separate measurement
%     - The FFT length equals the number of samples in the data
%
%     LMS Mode (default):
%     - Uses least squares fitting to extract harmonics (similar to tomdec)
%     - Fits harmonics up to specified order using sine/cosine basis
%     - Calculates noise floor from fitting residual
%     - If OSR > 1, applies window/filter/undo workflow to residual:
%       * Applies window with RMS normalization to preserve noise power
%       * Filters residual to signal bandwidth [0, Fs/2/OSR] before noise calculation
%     - Displays residue error circle on polar plot (radius = residual error in dB)
%     - Harmonics outside noise circle indicate significant distortion
%     - Best for clean harmonic analysis with intuitive noise reference
%
%     FFT Mode:
%     - Performs coherent averaging with phase alignment across multiple runs
%     - Applies window function before FFT to reduce spectral leakage
%     - Window is normalized to preserve RMS power
%     - Parabolic interpolation provides sub-bin frequency accuracy
%     - Shows all frequency bins including noise
%     - Best for analyzing full spectrum content
%
%     Both modes:
%     - Fundamental signal is shown in red, harmonics in blue
%     - The polar plot shows phase (angle) and magnitude (radius)
%     - Radius is displayed in dB with 0 dB at the perimeter
%
%   See also: plotspec, alias, fft, polarplot, window, ifilter

p = inputParser;
validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
validScalarNonNeg = @(x) isnumeric(x) && isscalar(x) && (x >= 0);
validMode = @(x) ischar(x) && ismember(lower(x), {'fft', 'lms'});
validWindow = @(x) (ischar(x) && ismember(x, {'hann', 'rect'})) || isa(x, 'function_handle');
addOptional(p, 'harmonic', 5);
addOptional(p, 'maxSignal', max(max(sig))-min(min(sig)), validScalarPosNum);
addParameter(p, 'Fs', 1, validScalarPosNum);
addParameter(p, 'OSR', 1, validScalarPosNum);
addParameter(p, 'cutoff', 0, validScalarNonNeg);
addParameter(p, 'mode', 'LMS', validMode);
addParameter(p, 'window', 'hann', validWindow);
parse(p, varargin{:});

%%
harmonic = p.Results.harmonic;
maxSignal = p.Results.maxSignal;
Fs = p.Results.Fs;
OSR = p.Results.OSR;
cutoffFreq = p.Results.cutoff;
mode = lower(p.Results.mode);
windowFunc = p.Results.window;

%%

[N,M] = size(sig);
if(M==1 && N > 1)
    sig = sig';
    N_fft = N;
else
    N_fft = M;
end

[N_run,~] = size(sig);

if strcmp(mode, 'fft')
    % ========== FFT Mode: Traditional coherent averaging ==========
    Nd2 = floor(N_fft/2/OSR);

    % Generate window function
    if ischar(windowFunc)
        % Use embedded window functions (no toolbox required)
        if strcmp(windowFunc, 'hann')
            win = hannwin(N_fft);
        elseif strcmp(windowFunc, 'rect')
            win = ones(1, N_fft);
        else
            win = ones(1, N_fft);
            warning('plotphase:invalidWindow', 'Unknown window type ''%s'', using rectangular window', windowFunc);
        end
    else
        % Use function handle (requires Signal Processing Toolbox)
        try
            win = window(windowFunc, N_fft, 'periodic')';
        catch
            try
                win = window(windowFunc, N_fft)';
            catch
                win = ones(1, N_fft);
                warning('plotphase:windowFailed', 'Failed to generate window, using rectangular window');
            end
        end
    end

    spec = zeros([1,N_fft]);
    ME = 0;
    for iter = 1:N_run
        tdata = sig(iter,:);
        if(rms(tdata)==0)
            continue;
        end
        tdata = tdata./maxSignal;
        tdata = tdata-mean(tdata);
        tdata = tdata.*win/sqrt(mean(win.^2));

    tspec = fft(tdata);
    tspec(1) = 0;
    [~, bin] = max(abs(tspec(1:floor(N_fft/2/OSR))));
    % Guard against bin = 1 (DC bin) which would cause division by zero
    if bin == 1
        warning('Signal detected at DC bin, skipping phase analysis for this run');
        continue;
    end

    % Refine bin location using parabolic interpolation for sub-bin accuracy
    tspec_mag = abs(tspec);
    sig_e = log10(tspec_mag(bin));
    sig_l = log10(tspec_mag(min(max(bin-1,1),floor(N_fft/2/OSR))));
    sig_r = log10(tspec_mag(min(max(bin+1,1),floor(N_fft/2/OSR))));
    bin_r = bin + (sig_r-sig_l)/(2*sig_e-sig_l-sig_r)/2;
    if(isnan(bin_r))
        bin_r = bin;
    end

    phi = tspec(bin)/abs(tspec(bin));

    phasor = conj(phi);
    marker = zeros(1,N_fft);
    for iter2 = 1:N_fft                         % harmonic phase shift
        J = (bin-1)*iter2;
        if(mod(floor(J/N_fft*2),2) == 0)
            b = J-floor(J/N_fft)*N_fft+1;
            if(marker(b) == 0)
                tspec(b) = tspec(b).*phasor;
                marker(b) = 1;
            end
        else
            b = N_fft-J+floor(J/N_fft)*N_fft+1;
            if(marker(b) == 0)
                tspec(b) = tspec(b).*conj(phasor);
                marker(b) = 1;
            end
        end
        phasor = phasor * conj(phi);
    end

    for iter2 = 1:N_fft                         % non-harmonic shift
        if(marker(iter2) == 0)
            tspec(iter2) = tspec(iter2).*(conj(phi).^((iter2-1)/(bin_r-1)));
        end
    end

    spec = spec + tspec;

    ME = ME+1;
end

spec = spec(1:Nd2);

% Remove flicker noise (1/f noise) if requested
if cutoffFreq > 0
    spec(1:ceil(cutoffFreq/Fs*N_fft)) = 0;
end

phi = spec./abs(spec);
spec = 10*log10(abs(spec).^2/(N_fft^2)*16/ME^2);
spec_sort = sort(spec);
minR = spec_sort(ceil(length(spec_sort)*0.01));
if(isinf(minR))
    minR = -100;
end
spec = max(spec,minR) - minR;
[~, bin] = max(spec);

% Refine bin location using parabolic interpolation for sub-bin accuracy
sig_e = spec(bin);
sig_l = spec(min(max(bin-1,1),Nd2));
sig_r = spec(min(max(bin+1,1),Nd2));
% Parabolic interpolation for sub-bin frequency accuracy
bin_r = bin + (sig_r-sig_l)/(2*sig_e-sig_l-sig_r)/2;
if(isnan(bin_r))
    bin_r = bin;
end

    spec = phi.*spec;

    % ========== FFT Mode Plotting ==========
    h = polarplot(spec,'k.');
    grid on;
    hold on;
    pax = gca;
    pax.ThetaDir = 'clockwise';
    pax.ThetaZeroLocation = 'top';
    tick = [-minR:-10:0];
    pax.RTick = tick(end:-1:1);
    tickl = (0:-10:minR);
    pax.RTickLabel = tickl(end:-1:1);
    rlim([0,-minR]);

    polarplot(spec(bin),'ro');
    polarplot([0,spec(bin)],'r-','linewidth',2);
    for iter = 2:harmonic
        b = alias(round((bin_r-1)*iter),N_fft);
        polarplot(spec(b+1),'bs');
        polarplot([0,spec(b+1)],'b-','linewidth',2);
        text(angle(spec(b+1))+0.1,abs(spec(b+1)),num2str(iter),'fontname','Arial','fontsize',8,'horizontalalignment','center');
    end

    title('Spectrum Phase (FFT)');

else
    % ========== LMS Mode: Least squares harmonic fitting ==========

    % Average all runs first
    sig_avg = mean(sig, 1);

    % Normalize signal
    sig_avg = sig_avg - mean(sig_avg);
    sig_avg = sig_avg / maxSignal;

    % Find fundamental frequency using FFT
    [~,freq] = sinfit(sig_avg);

    % Build sine/cosine basis for harmonics (similar to tomdec)
    t = 0:(N_fft-1);
    SI = zeros([N_fft, harmonic]);
    SQ = zeros([N_fft, harmonic]);
    for ii = 1:harmonic
        SI(:,ii) = cos(t*freq*ii*2*pi)';
        SQ(:,ii) = sin(t*freq*ii*2*pi)';
    end

    % Least squares fit
    W = linsolve([SI,SQ], sig_avg');

    % Reconstruct signal with all harmonics
    signal_all = [SI,SQ] * W;

    % Calculate residual (noise)
    residual = sig_avg' - signal_all;

    % Filter residual to signal bandwidth if oversampling
    if OSR > 1
        % Generate window function for residual
        N_residual = length(residual);
        if ischar(windowFunc)
            % Use embedded window functions (no toolbox required)
            if strcmp(windowFunc, 'hann')
                win = hannwin(N_residual);
            elseif strcmp(windowFunc, 'rect')
                win = ones(1, N_residual);
            else
                win = ones(1, N_residual);
                warning('plotphase:invalidWindow', 'Unknown window type ''%s'', using rectangular window', windowFunc);
            end
        else
            % Use function handle (requires Signal Processing Toolbox)
            try
                win = window(windowFunc, N_residual, 'periodic')';
            catch
                try
                    win = window(windowFunc, N_residual)';
                catch
                    win = ones(1, N_residual);
                    warning('plotphase:windowFailed', 'Failed to generate window, using rectangular window');
                end
            end
        end

        % Apply window with RMS normalization to preserve power
        residual = residual .* (win / sqrt(mean(win.^2)))';

        % Apply ideal low-pass filter
        residual = ifilter(residual, [0, 0.5/OSR]);

    end

    noise_power = rms(residual)*2*sqrt(2);
    noise_dB = 20*log10(noise_power);

    % Extract magnitude and phase for each harmonic
    harm_mag = zeros(harmonic,1);
    harm_phase = zeros(harmonic,1);
    for ii = 1:harmonic
        I_weight = W(ii);
        Q_weight = W(ii+harmonic);
        harm_mag(ii) = sqrt(I_weight^2 + Q_weight^2)*2;
        harm_phase(ii) = atan2(Q_weight, I_weight);
    end

    % Phase rotation: make phases relative to fundamental
    % Each harmonic's phase = phase_of_harmonic - (phase_of_fundamental * harmonic_order)
    fundamental_phase = harm_phase(1);
    for ii = 1:harmonic
        harm_phase(ii) = harm_phase(ii) - fundamental_phase * ii;
    end
    % Wrap phases to [-pi, pi]
    harm_phase = mod(harm_phase + pi, 2*pi) - pi;

    % Convert to dB (relative to full scale)
    harm_dB = 20*log10(harm_mag);

    % Set maxR and minR for plot scaling
    % Round maximum harmonic to nearest 10 dB
    maxR = ceil(max(harm_dB)/10)*10;
    % Set minimum to accommodate noise floor with margin
    minR = min(min(harm_dB), noise_dB) - 10;
    % Round minR to nearest 10 dB
    minR = floor(minR/10)*10;

    % ========== LMS Mode Plotting ==========
    % Create polar plot
    h = polarplot([0,0],[0,1],'w.');  % Dummy plot to initialize
    hold on;
    grid on;
    pax = gca;
    pax.ThetaDir = 'clockwise';
    pax.ThetaZeroLocation = 'top';

    % Draw noise circle
    theta_circle = linspace(0, 2*pi, 100);
    noise_radius = noise_dB - minR;  % Convert to plot scale
    polarplot(theta_circle, noise_radius*ones(size(theta_circle)), 'k--', 'LineWidth', 1.5);
    text(pi*3/4, noise_radius, sprintf('Residue Errors\n%.1f dB', noise_dB), ...
        'FontSize', 9, 'Color', 'k', 'HorizontalAlignment', 'left');

    % Plot harmonics
    for ii = 1:harmonic
        harm_radius = harm_dB(ii) - minR;  % Convert to plot scale
        if ii == 1
            % Fundamental in red
            polarplot(harm_phase(ii), harm_radius, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
            polarplot([0, harm_phase(ii)], [0, harm_radius], 'r-', 'LineWidth', 2);
        else
            % Harmonics in blue
            polarplot(harm_phase(ii), harm_radius, 'bs', 'MarkerSize', 6, 'LineWidth', 1.5);
            polarplot([0, harm_phase(ii)], [0, harm_radius], 'b-', 'LineWidth', 1.5);
            text(harm_phase(ii)+0.1, harm_radius, num2str(ii), ...
                'FontName', 'Arial', 'FontSize', 10, 'HorizontalAlignment', 'center');
        end
    end

    % Set axis limits
    tick = [0:10:(maxR-minR)];
    pax.RTick = tick;
    tickl = (minR:10:maxR);
    pax.RTickLabel = tickl;
    rlim([0, maxR-minR]);

    title('Signal Component Phase (LMS)');

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