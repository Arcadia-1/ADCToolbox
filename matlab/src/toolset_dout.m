function status = toolset_dout(bits, outputDir, varargin)
%TOOLSET_DOUT Run 6 digital analysis tools on ADC digital output
%
%   status = toolset_dout(bits, outputDir)
%   status = toolset_dout(bits, outputDir, 'Visible', true)
%
% Inputs:
%   bits       - Digital bits (N samples x B bits, MSB to LSB)
%   outputDir  - Directory to save output figures
%
% Optional Parameters:
%   'Visible'   - Show figures (default: false)
%   'Order'     - Polynomial order for calibration (default: 5)
%   'Prefix'    - Filename prefix (default: 'dout')
%
% Outputs:
%   status - Struct with fields:
%            .success (true if all tools completed)
%            .tools_completed (1x6 array of success flags)
%            .errors (cell array of error messages)
%            .panel_path (path to panel figure)
%
% Example:
%   bits = readmatrix('dout_SAR_12b.csv');
%   status = toolset_dout(bits, 'output/test2');

% Parse inputs
p = inputParser;
addParameter(p, 'Visible', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Order', 5, @isnumeric);
addParameter(p, 'Prefix', 'dout', @ischar);
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
status.tools_completed = zeros(1, 6);
status.errors = {};
status.panel_path = '';

% Create output directory
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

fprintf('\n=== Running DOUT Toolset (6 Tools) ===\n');

% Validate input data
fprintf('[Validation]');
try
    validateDoutData(bits);
    fprintf(' ✓\n');
catch ME
    fprintf(' ✗ %s\n', ME.message);
    error('Input validation failed: %s', ME.message);
end

nBits = size(bits, 2);
fprintf('Resolution: %d bits\n', nBits);

nominalWeights = 2.^(nBits-1:-1:0);

fprintf('[1/6][Spectrum (Nominal)]');
try
    digitalCodes_nominal = bits * nominalWeights';
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [ENoB_nom, SNDR_nom, SFDR_nom, SNR_nom, THD_nom, ~, ~, ~] = ...
        specPlot(digitalCodes_nominal, 'label', 1, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
    title('Digital Spectrum: Nominal Weights');
    set(gca, "FontSize", 16);
    pngPath = fullfile(outputDir, sprintf('%s_1_spectrum_nominal.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(1) = 1;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 1: %s', ME.message);
end

fprintf('[2/6][Spectrum (Calibrated)]');
try
    [weight_cal, ~, ~, ~, ~, freqCal] = FGCalSine(bits, 'freq', 0, 'order', opts.Order);
    digitalCodes_calibrated = bits * weight_cal';
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [ENoB_cal, SNDR_cal, SFDR_cal, SNR_cal, THD_cal, ~, ~, ~] = ...
        specPlot(digitalCodes_calibrated, 'label', 1, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
    title('Digital Spectrum: Calibrated Weights');
    set(gca, "FontSize", 16);
    pngPath = fullfile(outputDir, sprintf('%s_2_spectrum_calibrated.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(2) = 1;
    fprintf(' ✓ (+%.2f ENoB) → [%s]\n', ENoB_cal - ENoB_nom, pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 2: %s', ME.message);
end

fprintf('[3/6][Bit Activity]');
try
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    bit_usage = bitActivity(bits, 'AnnotateExtremes', true);
    title('Bit Activity Analysis');
    set(gca, "FontSize", 16);
    pngPath = fullfile(outputDir, sprintf('%s_3_bitActivity.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(3) = 1;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 3: %s', ME.message);
end

fprintf('[4/6][Overflow Check]');
try
    if ~exist('weight_cal', 'var')
        [weight_cal, ~, ~, ~, ~, ~] = FGCalSine(bits, 'freq', 0, 'order', opts.Order);
    end
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    data_decom = overflowChk(bits, weight_cal);
    title('Overflow Check');
    set(gca, "FontSize", 16);
    pngPath = fullfile(outputDir, sprintf('%s_4_overflowChk.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(4) = 1;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 4: %s', ME.message);
end

fprintf('[5/6][Weight Scaling]');
try
    if ~exist('weight_cal', 'var')
        [weight_cal, ~, ~, ~, ~, ~] = FGCalSine(bits, 'freq', 0, 'order', opts.Order);
    end
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    radix = weightScaling(weight_cal);
    title('Weight Scaling Analysis');
    set(gca, "FontSize", 16);
    pngPath = fullfile(outputDir, sprintf('%s_5_weightScaling.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(5) = 1;
    fprintf(' ✓ → [%s]\n', pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 5: %s', ME.message);
end

fprintf('[6/6][ENoB Bit Sweep]');
try
    if ~exist('freqCal', 'var')
        [~, ~, ~, ~, ~, freqCal] = FGCalSine(bits, 'freq', 0, 'order', opts.Order);
    end
    figure('Position', [100, 100, 800, 600], 'Visible', figVis);
    [ENoB_sweep, nBits_vec] = ENoB_bitSweep(bits, 'freq', freqCal, ...
        'order', opts.Order, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
    title('ENoB vs Number of Bits');
    set(gca, "FontSize", 16);
    pngPath = fullfile(outputDir, sprintf('%s_6_ENoB_sweep.png', opts.Prefix));
    saveas(gcf, pngPath);
    close(gcf);
    status.tools_completed(6) = 1;
    fprintf(' ✓ (Max: %.2f ENoB) → [%s]\n', max(ENoB_sweep), pngPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Tool 6: %s', ME.message);
end

fprintf('[Panel]');
try
    plotFiles = {
        fullfile(outputDir, sprintf('%s_1_spectrum_nominal.png', opts.Prefix));
        fullfile(outputDir, sprintf('%s_2_spectrum_calibrated.png', opts.Prefix));
        fullfile(outputDir, sprintf('%s_3_bitActivity.png', opts.Prefix));
        fullfile(outputDir, sprintf('%s_4_overflowChk.png', opts.Prefix));
        fullfile(outputDir, sprintf('%s_5_weightScaling.png', opts.Prefix));
        fullfile(outputDir, sprintf('%s_6_ENoB_sweep.png', opts.Prefix));
    };

    plotLabels = {
        '(1) Nominal Weights';
        '(2) Calibrated Weights';
        '(3) Bit Activity';
        '(4) Overflow Check';
        '(5) Weight Scaling';
        '(6) ENoB Bit Sweep';
    };

    fig = figure('Position', [50, 50, 1600, 1800], 'Visible', figVis);
    tlo = tiledlayout(fig, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

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
            axis([0, 1, 0, 1]);
            axis off;
            title(plotLabels{p}, 'FontSize', 12, 'Interpreter', 'none', 'Color', 'red');
        end
    end

    sgtitle('DOUT Toolset Overview', 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

    panelPath = fullfile(outputDir, sprintf('PANEL_%s.png', upper(opts.Prefix)));
    exportgraphics(fig, panelPath, 'Resolution', 600);
    close(fig);
    status.panel_path = panelPath;
    fprintf(' ✓ → [%s]\n', panelPath);
catch ME
    fprintf(' ✗ %s\n', ME.message);
    status.errors{end+1} = sprintf('Panel: %s', ME.message);
end

n_success = sum(status.tools_completed);
fprintf('=== Toolset complete: %d/6 tools succeeded ===\n\n', n_success);
status.success = (n_success == 6);

end
