function results = toolset_aout(postCal, freqCal, outputDir, varargin)
%TOOLSET_AOUT Run 9 analog analysis tools on calibrated ADC data
%
%   results = toolset_aout(postCal, freqCal, outputDir, options)
%
% Inputs:
%   postCal    - Calibrated ADC output signal (1D array)
%   freqCal    - Calibrated frequency (normalized Fin/Fs)
%   outputDir  - Directory to save output figures
%   options    - Optional name-value pairs:
%                'SaveFigs' (default: true) - Save individual figures
%                'CreatePanel' (default: true) - Create overview panel
%                'Visible' (default: false) - Figure visibility
%                'Resolution' (default: 11) - ADC resolution
%                'FullScale' (default: auto) - ADC full scale range
%
% Outputs:
%   results    - Struct containing all analysis results

% Parse optional inputs
p = inputParser;
addParameter(p, 'SaveFigs', true, @islogical);
addParameter(p, 'CreatePanel', true, @islogical);
addParameter(p, 'Visible', false, @islogical);
addParameter(p, 'Resolution', 11, @isnumeric);
addParameter(p, 'FullScale', max(postCal) - min(postCal), @isnumeric);
parse(p, varargin{:});
opts = p.Results;

% Set figure visibility
if opts.Visible
    figVis = 'on';
else
    figVis = 'off';
end

% Initialize results structure
results = struct();

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fprintf('\n=== Running AOUT Toolset (9 Analog Analysis Tools) ===\n');

% -------------------------------------------------------------------------
% Tool 1: tomDecomp - Time-domain Error Decomposition
% -------------------------------------------------------------------------
fprintf('  (1/9) Running tomDecomp (Time-domain Error Decomposition)...\n');
try
    figure('Position', [100, 100, 1200, 800], 'Visible', figVis);
    [signal, error, indep, dep, phi] = tomDecomp(postCal, freqCal, 10, 1);
    sgtitle('tomDecomp: Time-domain Error Decomposition');

    if opts.SaveFigs
        saveas(gcf, fullfile(outputDir, 'aout_1_tomDecomp.png'));
        fprintf('      [Saved] aout_1_tomDecomp.png\n');
    end
    close(gcf);

    results.tomDecomp.signal = signal;
    results.tomDecomp.error = error;
    results.tomDecomp.indep = indep;
    results.tomDecomp.dep = dep;
    results.tomDecomp.phi = phi;
    results.tomDecomp.rms_error = rms(error);
    fprintf('      phi=%.4f rad, rms_error=%.6f\n', phi, rms(error));
catch ME
    fprintf('      [Error] %s\n', ME.message);
    results.tomDecomp.error_msg = ME.message;
end

% -------------------------------------------------------------------------
% Tool 2: specPlot - Frequency Spectrum
% -------------------------------------------------------------------------
fprintf('  (2/9) Running specPlot (Frequency Spectrum)...\n');
try
    figure('Position', [100, 100, 1200, 800], 'Visible', figVis);
    [ENoB, SNDR, SFDR, SNR, THD, pwr, NF, ~] = specPlot(postCal, 'label', 1, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
    title('specPlot: Frequency Spectrum');

    if opts.SaveFigs
        saveas(gcf, fullfile(outputDir, 'aout_2_specPlot.png'));
        fprintf('      [Saved] aout_2_specPlot.png\n');
    end
    close(gcf);

    results.specPlot.ENoB = ENoB;
    results.specPlot.SNDR = SNDR;
    results.specPlot.SFDR = SFDR;
    results.specPlot.SNR = SNR;
    results.specPlot.THD = THD;
    results.specPlot.pwr = pwr;
    results.specPlot.NF = NF;
    fprintf('      ENoB=%.2f, SNDR=%.2f dB, SFDR=%.2f dB\n', ENoB, SNDR, SFDR);
catch ME
    fprintf('      [Error] %s\n', ME.message);
    results.specPlot.error_msg = ME.message;
end

% -------------------------------------------------------------------------
% Tool 3: specPlotPhase - Phase-domain Error
% -------------------------------------------------------------------------
fprintf('  (3/9) Running specPlotPhase (Phase-domain Error)...\n');
try
    figure('Position', [100, 100, 1200, 800], 'Visible', figVis);
    [h, spec, phi, bin] = specPlotPhase(postCal, 'harmonic', 10);
    title('specPlotPhase: Phase-domain Error');

    if opts.SaveFigs
        saveas(gcf, fullfile(outputDir, 'aout_3_specPlotPhase.png'));
        fprintf('      [Saved] aout_3_specPlotPhase.png\n');
    end
    close(gcf);

    results.specPlotPhase.h = h;
    results.specPlotPhase.spec = spec;
    results.specPlotPhase.phi = phi;
    results.specPlotPhase.bin = bin;
    fprintf('      fundamental bin: %d\n', bin);
catch ME
    fprintf('      [Error] %s\n', ME.message);
    results.specPlotPhase.error_msg = ME.message;
end

% Compute error data using sineFit for error-based tools
fprintf('  Computing error data using sineFit...\n');
try
    [data_fit, freq_est, mag, dc, phi] = sineFit(postCal);
    err_data = postCal - data_fit;
    results.sineFit.freq_est = freq_est;
    results.sineFit.mag = mag;
    results.sineFit.dc = dc;
    results.sineFit.phi = phi;
    fprintf('      freq_est=%.6f, mag=%.6f, dc=%.6f\n', freq_est, mag, dc);
catch ME
    fprintf('      [Error] Cannot compute sineFit: %s\n', ME.message);
    err_data = postCal - mean(postCal);  % Fallback
    results.sineFit.error_msg = ME.message;
end

% -------------------------------------------------------------------------
% Tool 4: errHistSine (code mode) - Error Histogram by Code
% -------------------------------------------------------------------------
fprintf('  (4/9) Running errHistSine - by code (Error Histogram by Code)...\n');
try
    figure('Position', [100, 100, 1200, 800], 'Visible', figVis);
    [emean_code, erms_code, code_axis, ~, ~, ~, ~] = errHistSine(postCal, 'bin', 20, 'fin', freqCal, 'disp', 1, 'mode', 1);
    sgtitle('errHistSine (code): Error Histogram by Code');

    if opts.SaveFigs
        saveas(gcf, fullfile(outputDir, 'aout_4_errHistSine_code.png'));
        fprintf('      [Saved] aout_4_errHistSine_code.png\n');
    end
    close(gcf);

    results.errHistSine_code.emean = emean_code;
    results.errHistSine_code.erms = erms_code;
    results.errHistSine_code.code_axis = code_axis;
catch ME
    fprintf('      [Error] %s\n', ME.message);
    results.errHistSine_code.error_msg = ME.message;
end

% -------------------------------------------------------------------------
% Tool 5: errHistSine (phase mode) - Error Histogram by Phase
% -------------------------------------------------------------------------
fprintf('  (5/9) Running errHistSine - by phase (Error Histogram by Phase)...\n');
try
    figure('Position', [100, 100, 1200, 800], 'Visible', figVis);
    [emean, erms, phase_code, anoi, pnoi, ~, ~] = errHistSine(postCal, 'bin', 99, 'fin', freqCal, 'disp', 1, 'mode', 0);
    sgtitle('errHistSine (phase): Error Histogram by Phase');

    if opts.SaveFigs
        saveas(gcf, fullfile(outputDir, 'aout_5_errHistSine_phase.png'));
        fprintf('      [Saved] aout_5_errHistSine_phase.png\n');
    end
    close(gcf);

    results.errHistSine_phase.emean = emean;
    results.errHistSine_phase.erms = erms;
    results.errHistSine_phase.phase_code = phase_code;
    results.errHistSine_phase.anoi = anoi;
    results.errHistSine_phase.pnoi = pnoi;
    fprintf('      anoi=%.6f, pnoi=%.6f rad\n', anoi, pnoi);
catch ME
    fprintf('      [Error] %s\n', ME.message);
    results.errHistSine_phase.error_msg = ME.message;
end

% -------------------------------------------------------------------------
% Tool 6: errPDF - Error PDF
% -------------------------------------------------------------------------
fprintf('  (6/9) Running errPDF (Error PDF)...\n');
try
    figure('Position', [100, 100, 1200, 800], 'Visible', figVis);
    [~, mu, sigma, KL_divergence, x, fx, gauss_pdf] = errPDF(err_data, ...
        'Resolution', opts.Resolution, 'FullScale', opts.FullScale);
    title('errPDF: Error PDF');

    if opts.SaveFigs
        saveas(gcf, fullfile(outputDir, 'aout_6_errPDF.png'));
        fprintf('      [Saved] aout_6_errPDF.png\n');
    end
    close(gcf);

    results.errPDF.mu = mu;
    results.errPDF.sigma = sigma;
    results.errPDF.KL_divergence = KL_divergence;
    results.errPDF.x = x;
    results.errPDF.fx = fx;
    results.errPDF.gauss_pdf = gauss_pdf;
    fprintf('      mu=%.4f, sigma=%.4f, KL=%.6f\n', mu, sigma, KL_divergence);
catch ME
    fprintf('      [Error] %s\n', ME.message);
    results.errPDF.error_msg = ME.message;
end

% -------------------------------------------------------------------------
% Tool 7: errAutoCorrelation - Error Autocorrelation
% -------------------------------------------------------------------------
fprintf('  (7/9) Running errAutoCorrelation (Error Autocorrelation)...\n');
try
    figure('Position', [100, 100, 1200, 800], 'Visible', figVis);
    [acf, lags] = errAutoCorrelation(err_data, 'MaxLag', 200, 'Normalize', true);
    title('errAutoCorrelation: Error Autocorrelation');

    if opts.SaveFigs
        saveas(gcf, fullfile(outputDir, 'aout_7_errAutoCorrelation.png'));
        fprintf('      [Saved] aout_7_errAutoCorrelation.png\n');
    end
    close(gcf);

    results.errAutoCorrelation.acf = acf;
    results.errAutoCorrelation.lags = lags;
catch ME
    fprintf('      [Error] %s\n', ME.message);
    results.errAutoCorrelation.error_msg = ME.message;
end

% -------------------------------------------------------------------------
% Tool 8: Error Spectrum - specPlot on error data
% -------------------------------------------------------------------------
fprintf('  (8/9) Running specPlot on error (Error Spectrum)...\n');
try
    figure('Position', [100, 100, 1200, 800], 'Visible', figVis);
    [~, ~, ~, ~, ~, ~, ~, h] = specPlot(err_data, 'label', 0);
    title('errSpectrum: Error Spectrum');

    if opts.SaveFigs
        saveas(gcf, fullfile(outputDir, 'aout_8_errSpectrum.png'));
        fprintf('      [Saved] aout_8_errSpectrum.png\n');
    end
    close(gcf);

    results.errSpectrum.h = h;
catch ME
    fprintf('      [Error] %s\n', ME.message);
    results.errSpectrum.error_msg = ME.message;
end

% -------------------------------------------------------------------------
% Tool 9: errEnvelopeSpectrum - Error Envelope Spectrum
% -------------------------------------------------------------------------
fprintf('  (9/9) Running errEnvelopeSpectrum (Error Envelope Spectrum)...\n');
try
    figure('Position', [100, 100, 1200, 800], 'Visible', figVis);
    errEnvelopeSpectrum(err_data, 'Fs', 1);
    title('errEnvelopeSpectrum: Error Envelope Spectrum');

    if opts.SaveFigs
        saveas(gcf, fullfile(outputDir, 'aout_9_errEnvelopeSpectrum.png'));
        fprintf('      [Saved] aout_9_errEnvelopeSpectrum.png\n');
    end
    close(gcf);

    results.errEnvelopeSpectrum.completed = true;
catch ME
    fprintf('      [Error] %s\n', ME.message);
    results.errEnvelopeSpectrum.error_msg = ME.message;
end

% -------------------------------------------------------------------------
% Create Panel Overview (3x3 grid)
% -------------------------------------------------------------------------
if opts.CreatePanel
    fprintf('\n  Creating AOUT Panel Overview (3x3 grid)...\n');

    plotFiles = {
        fullfile(outputDir, 'aout_1_tomDecomp.png');
        fullfile(outputDir, 'aout_2_specPlot.png');
        fullfile(outputDir, 'aout_3_specPlotPhase.png');
        fullfile(outputDir, 'aout_4_errHistSine_code.png');
        fullfile(outputDir, 'aout_5_errHistSine_phase.png');
        fullfile(outputDir, 'aout_6_errPDF.png');
        fullfile(outputDir, 'aout_7_errAutoCorrelation.png');
        fullfile(outputDir, 'aout_8_errSpectrum.png');
        fullfile(outputDir, 'aout_9_errEnvelopeSpectrum.png');
    };

    plotLabels = {
        '(1) Time-domain Error Decomposition';
        '(2) Frequency Spectrum';
        '(3) Phase-domain Error';
        '(4) Error Histogram by Code';
        '(5) Error Histogram by Phase';
        '(6) Error PDF';
        '(7) Error Autocorrelation';
        '(8) Error Spectrum';
        '(9) Error Envelope Spectrum';
    };

    fig = figure('Position', [50 50 1800 1800], 'Visible', figVis);
    tlo = tiledlayout(fig, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    for p = 1:length(plotFiles)
        nexttile(tlo, p);
        img_path = plotFiles{p};

        try
            if isfile(img_path)
                img = imread(img_path);
                imshow(img, 'Border', 'tight');
                axis tight;
                axis off;
                title(plotLabels{p}, 'FontSize', 12, 'Interpreter', 'none');
            else
                text(0.5, 0.5, sprintf('Missing:\n%s', plotLabels{p}), ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'FontSize', 10, 'Color', 'red');
                axis([0 1 0 1]);
                axis off;
                title(plotLabels{p}, 'FontSize', 12, 'Interpreter', 'none', 'Color', 'red');
            end
        catch ME
            text(0.5, 0.5, sprintf('Error:\n%s', ME.message), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', 10, 'Color', 'red');
            axis([0 1 0 1]);
            axis off;
            title(plotLabels{p}, 'FontSize', 12, 'Interpreter', 'none', 'Color', 'red');
        end
    end

    timeStr = datestr(now, 'yyyymmdd_HHMMSS');
    sgtitle(sprintf('AOUT Toolset Overview - %s', timeStr), 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

    panelPath = fullfile(outputDir, sprintf('PANEL_AOUT_%s.png', timeStr));
    exportgraphics(fig, panelPath, 'Resolution', 300);
    fprintf('    [Saved] %s\n', panelPath);
    close(fig);

    results.panel_path = panelPath;
end

fprintf('\n=== AOUT Toolset Complete ===\n\n');

end
