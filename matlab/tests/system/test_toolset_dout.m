%% test_toolset_dout.m - Run 6 digital analysis tools on ADC digital output
close all; clc; clear; warning("off")

%% Configuration
verbose = 0;
inputDir = "dataset";
outputDir = "test_output";
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'dout_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% ADC Configuration
Resolution = 10;  % ADC resolution in bits
Order = 5;        % Polynomial order for calibration

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    % Read digital bits (binary matrix: N samples x B bits)
    bits = readmatrix(dataFilePath);

    % Get dataset name and create subfolder
    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);

    % Determine actual resolution from data
    nBits = size(bits, 2);
    fprintf('  Resolution: %d bits\n', nBits);

    % Calculate nominal weights (2^(n-1), 2^(n-2), ..., 2^0)
    nominalWeights = 2.^(nBits-1:-1:0);

    % =========================================================================
    % Tool 1: Digital Spectrum with Nominal Weights
    % =========================================================================
    try
        digitalCodes_nominal = bits * nominalWeights';

        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        [ENoB_nom, SNDR_nom, SFDR_nom, SNR_nom, THD_nom, ~, ~, ~] = ...
            specPlot(digitalCodes_nominal, 'label', 1, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
        title(sprintf('Nominal Weights: %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "dout_1_spectrum_nominal.png", verbose);
    catch ME
        fprintf('    [Error] Nominal spectrum: %s\n', ME.message);
    end

    % =========================================================================
    % Tool 2: Digital Spectrum with Calibrated Weights
    % =========================================================================
    try
        % Auto-detect frequency using FGCalSine
        [weight_cal, ~, ~, ~, ~, freqCal] = FGCalSine(bits, 'freq', 0, 'order', Order);
        digitalCodes_calibrated = bits * weight_cal';

        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        [ENoB_cal, SNDR_cal, SFDR_cal, SNR_cal, THD_cal, ~, ~, ~] = ...
            specPlot(digitalCodes_calibrated, 'label', 1, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
        title(sprintf('Calibrated Weights: %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "dout_2_spectrum_calibrated.png", verbose);
    catch ME
        fprintf('    [Error] Calibrated spectrum: %s\n', ME.message);
    end

    % =========================================================================
    % Tool 3: Bit Activity
    % =========================================================================
    try
        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        bit_usage = bitActivity(bits, 'AnnotateExtremes', true);
        title(sprintf('Bit Activity: %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "dout_3_bitActivity.png", verbose);
    catch ME
        fprintf('    [Error] Bit activity: %s\n', ME.message);
    end

    % =========================================================================
    % Tool 4: Overflow Check
    % =========================================================================
    try
        if ~exist('weight_cal', 'var')
            [weight_cal, ~, ~, ~, ~, ~] = FGCalSine(bits, 'freq', 0, 'order', Order);
        end

        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        data_decom = overflowChk(bits, weight_cal);
        title(sprintf('Overflow Check: %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "dout_4_overflowChk.png", verbose);
    catch ME
        fprintf('    [Error] Overflow check: %s\n', ME.message);
    end

    % =========================================================================
    % Tool 5: Weight Scaling
    % =========================================================================
    try
        if ~exist('weight_cal', 'var')
            [weight_cal, ~, ~, ~, ~, ~] = FGCalSine(bits, 'freq', 0, 'order', Order);
        end

        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        radix = weightScaling(weight_cal);
        title(sprintf('Weight Scaling: %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "dout_5_weightScaling.png", verbose);
    catch ME
        fprintf('    [Error] Weight scaling: %s\n', ME.message);
    end

    % =========================================================================
    % Tool 6: ENoB Bit Sweep
    % =========================================================================
    try
        if ~exist('freqCal', 'var')
            [~, ~, ~, ~, ~, freqCal] = FGCalSine(bits, 'freq', 0, 'order', Order);
        end

        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        [ENoB_sweep, nBits_vec] = ENoB_bitSweep(bits, 'freq', freqCal, ...
            'order', Order, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
        title(sprintf('ENoB Bit Sweep: %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "dout_6_ENoB_sweep.png", verbose);

        fprintf('  Max ENoB: %.2f at %d bits\n', max(ENoB_sweep), nBits_vec(ENoB_sweep == max(ENoB_sweep)));
    catch ME
        fprintf('    [Error] ENoB sweep: %s\n', ME.message);
    end

    %%
    % =========================================================================
    % Create Panel Overview (3x2 grid)
    % =========================================================================

    plotFiles = {; ...
        fullfile(subFolder, 'dout_1_spectrum_nominal.png'); ...
        fullfile(subFolder, 'dout_2_spectrum_calibrated.png'); ...
        fullfile(subFolder, 'dout_3_bitActivity.png'); ...
        fullfile(subFolder, 'dout_4_overflowChk.png'); ...
        fullfile(subFolder, 'dout_5_weightScaling.png'); ...
        fullfile(subFolder, 'dout_6_ENoB_sweep.png'); ...
        };

    plotLabels = {; ...
        '(1) Nominal Weights'; ...
        '(2) Calibrated Weights'; ...
        '(3) Bit Activity'; ...
        '(4) Overflow Check'; ...
        '(5) Weight Scaling'; ...
        '(6) ENoB Bit Sweep'; ...
        };

    fig = figure('Position', [50, 50, 1600, 1800], 'Visible', verbose);
    tlo = tiledlayout(fig, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

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
                axis([0, 1, 0, 1]);
                axis off;
                title(plotLabels{p}, 'FontSize', 12, 'Interpreter', 'none', 'Color', 'red');
            end
        catch ME
            text(0.5, 0.5, sprintf('Error:\n%s', ME.message), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', 10, 'Color', 'red');
            axis([0, 1, 0, 1]);
            axis off;
            title(plotLabels{p}, 'FontSize', 12, 'Interpreter', 'none', 'Color', 'red');
        end
    end

    sgtitle(sprintf('DOUT Toolset: %s', datasetName), 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

    panelPath = fullfile(subFolder, 'PANEL_DOUT.png');
    exportgraphics(fig, panelPath, 'Resolution', 600);
    close(fig);
    fprintf('  [%s]->[%s]\n', mfilename, panelPath);

    fprintf('  [Done] %s\n\n', datasetName);
end