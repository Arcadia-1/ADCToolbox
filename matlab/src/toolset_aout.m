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
%            .plot_files (cell array of generated PNG paths)
%
% Example:
%   aout_data = readmatrix('sinewave_jitter.csv');
%   status = toolset_aout(aout_data, 'output/test1');
%   panel_status = toolset_aout_panel(status.plot_files, outputDir, 'Prefix', 'aout');

% Parse inputs
p = inputParser;
addParameter(p, 'Visible', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Resolution', 11, @isnumeric);
addParameter(p, 'Prefix', 'aout', @ischar);
parse(p, varargin{:});
opts = p.Results;

% Set figure visibility (accept 0/1 or true/false)
figVis = ternary(opts.Visible, 'on', 'off');

% Initialize status
status.success = false;
status.tools_completed = zeros(1, 9);
status.errors = {};
status.plot_files = cell(9, 1);

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

% Pre-compute common parameters
freqCal = findFin(aout_data);
FullScale = max(aout_data) - min(aout_data);

% Calculate error data for tools 6-9
try
    [data_fit, ~, ~, ~, ~] = sineFit(aout_data);
    err_data = aout_data - data_fit;
catch
    err_data = aout_data - mean(aout_data);
end

%% Tool 1: tomDecomp
fprintf('[1/9][tomDecomp]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [~, ~, ~, ~, ~] = tomDecomp(aout_data, freqCal, 10, 1);
    sgtitle('tomDecomp: Time-domain Error Decomposition', 'FontWeight', 'bold');
    set(gca, 'FontSize', 14);
    pngPath = fullfile(outputDir, sprintf('%s_1_tomDecomp.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(1) = 1;
    status.plot_files{1} = pngPath;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 1: %s', ME.message);
end

%% Tool 2: specPlot
fprintf('[2/9][specPlot]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [~, ~, ~, ~, ~, ~, ~, ~] = specPlot(aout_data, 'label', 1, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
    title('specPlot: Frequency Spectrum');
    set(gca, 'FontSize', 14);
    pngPath = fullfile(outputDir, sprintf('%s_2_specPlot.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(2) = 1;
    status.plot_files{2} = pngPath;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 2: %s', ME.message);
end

%% Tool 3: specPlotPhase
fprintf('[3/9][specPlotPhase]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [~, ~, ~, ~] = specPlotPhase(aout_data, 'harmonic', 10);
    title('specPlotPhase: Phase-domain Error');
    set(gca, 'FontSize', 14);
    pngPath = fullfile(outputDir, sprintf('%s_3_specPlotPhase.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(3) = 1;
    status.plot_files{3} = pngPath;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 3: %s', ME.message);
end

%% Tool 4: errHistSine (code)
fprintf('[4/9][errHistSine_code]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [~, ~, ~, ~, ~, ~, ~] = errHistSine(aout_data, 'bin', 20, 'fin', freqCal, 'disp', 1, 'mode', 1);
    sgtitle('errHistSine (code): Error Histogram by Code', 'FontWeight', 'bold');
    set(gca, 'FontSize', 14);
    pngPath = fullfile(outputDir, sprintf('%s_4_errHistSine_code.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(4) = 1;
    status.plot_files{4} = pngPath;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 4: %s', ME.message);
end

%% Tool 5: errHistSine (phase)
fprintf('[5/9][errHistSine_phase]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [~, ~, ~, ~, ~, ~, ~] = errHistSine(aout_data, 'bin', 99, 'fin', freqCal, 'disp', 1, 'mode', 0);
    sgtitle('errHistSine (phase): Error Histogram by Phase', 'FontWeight', 'bold');
    set(gca, 'FontSize', 14);
    pngPath = fullfile(outputDir, sprintf('%s_5_errHistSine_phase.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(5) = 1;
    status.plot_files{5} = pngPath;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 5: %s', ME.message);
end

%% Tool 6: errPDF
fprintf('[6/9][errPDF]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [~, ~, ~, ~, ~, ~, ~] = errPDF(err_data, 'Resolution', opts.Resolution, 'FullScale', FullScale);
    title('errPDF: Error PDF');
    set(gca, 'FontSize', 14);
    pngPath = fullfile(outputDir, sprintf('%s_6_errPDF.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(6) = 1;
    status.plot_files{6} = pngPath;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 6: %s', ME.message);
end

%% Tool 7: errAutoCorrelation
fprintf('[7/9][errAutoCorrelation]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [~, ~] = errAutoCorrelation(err_data, 'MaxLag', 200, 'Normalize', true);
    title('errAutoCorrelation: Error Autocorrelation');
    set(gca, 'FontSize', 14);
    pngPath = fullfile(outputDir, sprintf('%s_7_errAutoCorrelation.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(7) = 1;
    status.plot_files{7} = pngPath;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 7: %s', ME.message);
end

%% Tool 8: errSpectrum
fprintf('[8/9][errSpectrum]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [~, ~, ~, ~, ~, ~, ~, ~] = specPlot(err_data, 'label', 0);
    title('errSpectrum: Error Spectrum');
    set(gca, 'FontSize', 14);
    pngPath = fullfile(outputDir, sprintf('%s_8_errSpectrum.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(8) = 1;
    status.plot_files{8} = pngPath;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 8: %s', ME.message);
end

%% Tool 9: errEnvelopeSpectrum
fprintf('[9/9][errEnvelopeSpectrum]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    errEnvelopeSpectrum(err_data, 'Fs', 1);
    title('errEnvelopeSpectrum: Error Envelope Spectrum');
    set(gca, 'FontSize', 14);
    pngPath = fullfile(outputDir, sprintf('%s_9_errEnvelopeSpectrum.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(9) = 1;
    status.plot_files{9} = pngPath;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 9: %s', ME.message);
end

n_success = sum(status.tools_completed);
fprintf('=== Toolset complete: %d/9 tools succeeded ===\n\n', n_success);
status.success = (n_success == 9);
end

%% Helper Functions

function validateAoutData(data)
    % Validate aout_data is a numeric vector or 2D array
    if ~isnumeric(data) || isempty(data)
        error('aout_data must be a non-empty numeric array');
    end
    if ~ismatrix(data) || ndims(data) > 2
        error('aout_data must be a 1D vector or 2D array');
    end
end
