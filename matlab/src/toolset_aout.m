function status = toolset_aout(aout_data, outputDir, varargin)
%TOOLSET_AOUT Run 9 analog analysis tools on calibrated ADC data
%
%   status = toolset_aout(aout_data, outputDir)
%   status = toolset_aout(aout_data, outputDir, 'Visible', true)
%
% Inputs:
%   aout_data  - Analog output signal (1D vector)
%   outputDir  - Directory to save output figures
%
% Optional Parameters:
%   'Visible'   - Show figures (default: false)
%   'Resolution' - ADC resolution in bits (default: 11)
%   'Prefix'    - Filename prefix (default: 'aout')
%
% Outputs:
%   status - Struct with fields:
%            .success (true if all tools completed)
%            .tools_completed (1x9 array of success flags)
%            .errors (cell array of error messages)
%            .panel_path (path to panel figure)
%
% Example:
%   aout_data = readmatrix('sinewave_jitter.csv');
%   status = toolset_aout(aout_data, 'output/test1');

% Parse inputs
p = inputParser;
addParameter(p, 'Visible', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Resolution', 11, @isnumeric);
addParameter(p, 'Prefix', 'aout', @ischar);
parse(p, varargin{:});
opts = p.Results;

% Set figure visibility (accept 0/1 or true/false)
if opts.Visible
    figVis = 'on';
else
    figVis = 'off';
end

% Initialize status
status.success = false;
status.tools_completed = zeros(1, 9);
status.errors = {};
status.panel_path = '';

% Create output directory
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

% Validate input data
fprintf('[Validation]');
try
    validateAoutData(aout_data);
    fprintf(' ✓\n');
catch ME
    fprintf(' ✗ %s\n', ME.message);
    error('Input validation failed: %s', ME.message);
end

% Handle multirun data (take first row if 2D)
if size(aout_data, 1) > 1
    aout_data = aout_data(1, :);
end

freqCal = findFin(aout_data);
FullScale = max(aout_data) - min(aout_data);

fprintf('[1/9][tomDecomp]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [signal, error, indep, dep, phi] = tomDecomp(aout_data, freqCal, 10, 1);
    sgtitle('tomDecomp: Time-domain Error Decomposition');
    set(gca, "FontSize", 14);
    pngPath = fullfile(outputDir, sprintf('%s_1_tomDecomp.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(1) = 1;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 1: %s', ME.message);
end

fprintf('[2/9][specPlot]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [ENoB, SNDR, SFDR, SNR, THD, pwr, NF, ~] = specPlot(aout_data, 'label', 1, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
    title('specPlot: Frequency Spectrum');
    set(gca, "FontSize", 14);
    pngPath = fullfile(outputDir, sprintf('%s_2_specPlot.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(2) = 1;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 2: %s', ME.message);
end

fprintf('[3/9][specPlotPhase]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [h, spec, phi, bin] = specPlotPhase(aout_data, 'harmonic', 10);
    title('specPlotPhase: Phase-domain Error');
    set(gca, "FontSize", 14);
    pngPath = fullfile(outputDir, sprintf('%s_3_specPlotPhase.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(3) = 1;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 3: %s', ME.message);
end

try
    [data_fit, freq_est, mag, dc, phi] = sineFit(aout_data);
    err_data = aout_data - data_fit;
catch
    err_data = aout_data - mean(aout_data);
end

fprintf('[4/9][errHistSine (code)]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [emean_code, erms_code, code_axis, ~, ~, ~, ~] = errHistSine(aout_data, 'bin', 20, 'fin', freqCal, 'disp', 1, 'mode', 1);
    sgtitle('errHistSine (code): Error Histogram by Code');
    set(gca, "FontSize", 14);
    pngPath = fullfile(outputDir, sprintf('%s_4_errHistSine_code.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(4) = 1;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 4: %s', ME.message);
end

fprintf('[5/9][errHistSine (phase)]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [emean, erms, phase_code, anoi, pnoi, ~, ~] = errHistSine(aout_data, 'bin', 99, 'fin', freqCal, 'disp', 1, 'mode', 0);
    sgtitle('errHistSine (phase): Error Histogram by Phase');
    set(gca, "FontSize", 14);
    pngPath = fullfile(outputDir, sprintf('%s_5_errHistSine_phase.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(5) = 1;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 5: %s', ME.message);
end

fprintf('[6/9][errPDF]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [~, mu, sigma, KL_divergence, x, fx, gauss_pdf] = errPDF(err_data, ...
        'Resolution', opts.Resolution, 'FullScale', FullScale);
    title('errPDF: Error PDF');
    set(gca, "FontSize", 14);
    pngPath = fullfile(outputDir, sprintf('%s_6_errPDF.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(6) = 1;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 6: %s', ME.message);
end

fprintf('[7/9][errAutoCorrelation]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [acf, lags] = errAutoCorrelation(err_data, 'MaxLag', 200, 'Normalize', true);
    title('errAutoCorrelation: Error Autocorrelation');
    set(gca, "FontSize", 14);
    pngPath = fullfile(outputDir, sprintf('%s_7_errAutoCorrelation.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(7) = 1;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 7: %s', ME.message);
end

fprintf('[8/9][errSpectrum]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [~, ~, ~, ~, ~, ~, ~, h] = specPlot(err_data, 'label', 0);
    title('errSpectrum: Error Spectrum');
    set(gca, "FontSize", 14);
    pngPath = fullfile(outputDir, sprintf('%s_8_errSpectrum.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(8) = 1;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 8: %s', ME.message);
end

fprintf('[9/9][errEnvelopeSpectrum]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    errEnvelopeSpectrum(err_data, 'Fs', 1);
    title('errEnvelopeSpectrum: Error Envelope Spectrum');
    set(gca, "FontSize", 14);
    pngPath = fullfile(outputDir, sprintf('%s_9_errEnvelopeSpectrum.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(9) = 1;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 9: %s', ME.message);
end

fprintf('[Panel]');
try
    plotFiles = {
        fullfile(outputDir, sprintf('%s_1_tomDecomp.png', opts.Prefix));
        fullfile(outputDir, sprintf('%s_2_specPlot.png', opts.Prefix));
        fullfile(outputDir, sprintf('%s_3_specPlotPhase.png', opts.Prefix));
        fullfile(outputDir, sprintf('%s_4_errHistSine_code.png', opts.Prefix));
        fullfile(outputDir, sprintf('%s_5_errHistSine_phase.png', opts.Prefix));
        fullfile(outputDir, sprintf('%s_6_errPDF.png', opts.Prefix));
        fullfile(outputDir, sprintf('%s_7_errAutoCorrelation.png', opts.Prefix));
        fullfile(outputDir, sprintf('%s_8_errSpectrum.png', opts.Prefix));
        fullfile(outputDir, sprintf('%s_9_errEnvelopeSpectrum.png', opts.Prefix));
    };

    plotLabels = {
        '(1) tomDecomp';
        '(2) specPlot';
        '(3) specPlotPhase';
        '(4) errHistSine (code)';
        '(5) errHistSine (phase)';
        '(6) errPDF';
        '(7) errAutoCorrelation';
        '(8) errSpectrum';
        '(9) errEnvelopeSpectrum';
    };

    fig = figure('Position', [50 50 1800 1800], 'Visible', figVis);
    tlo = tiledlayout(fig, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    for p = 1:length(plotFiles)
        nexttile(tlo, p);
        img_path = plotFiles{p};

        if isfile(img_path)
            img = imread(img_path);
            imshow(img, 'Border', 'tight');
            axis tight;
            axis off;
            title(plotLabels{p}, 'FontSize', 12, 'Interpreter', 'none');
        else
            text(0.5, 0.5, sprintf('Missing:\n%s', plotLabels{p}), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 10, 'Color', 'red');
            axis([0 1 0 1]);
            axis off;
            title(plotLabels{p}, 'FontSize', 12, 'Interpreter', 'none', 'Color', 'red');
        end
    end

    sgtitle('AOUT Toolset Overview', 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

    panelPath = fullfile(outputDir, sprintf('PANEL_%s.png', upper(opts.Prefix)));
    saveas(fig, panelPath);
    close(fig);
    status.panel_path = panelPath;
    fprintf(' ✓ → [%s]\n', panelPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Panel: %s', ME.message);
end

n_success = sum(status.tools_completed);
fprintf('=== Toolset complete: %d/9 tools succeeded ===\n\n', n_success);
status.success = (n_success == 9);
end
