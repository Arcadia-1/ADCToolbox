%% test_toolset_aout.m - Run 9 analog analysis tools on calibrated ADC data
close all; clc; clear; warning("off")
%% Configuration
verbose = 0;
inputDir = "dataset";
outputDir = "test_output";
filesList = {"sinewave_INL_k2_0P0010_k3_0P0120.csv"};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv', 'batch_sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end
%% ADC Configuration
Resolution = 10; % ADC resolution in bits
%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    % Handle multirun data (take first row if 2D)
    if size(read_data, 1) > 1
        read_data = read_data(1, :);
    end

    % Get dataset name and create subfolder
    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);

    freqCal = findFin(read_data); % Estimate frequency using sineFit
    FullScale = max(read_data) - min(read_data); % FullScale estimation

    % =========================================================================
    % Tool 1: tomDecomp - Time-domain Error Decomposition
    % =========================================================================
    try
        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        [signal, error, indep, dep, phi] = tomDecomp(read_data, freqCal, 10, 1);
        sgtitle(sprintf('tomDecomp: %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "aout_1_tomDecomp.png", verbose);

    catch ME
        fprintf('    [Error] tomDecomp: %s\n', ME.message);
    end

    % =========================================================================
    % Tool 2: specPlot - Frequency Spectrum
    % =========================================================================
    try
        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        [ENoB, SNDR, SFDR, SNR, THD, pwr, NF, ~] = specPlot(read_data, 'label', 1, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);
        title(sprintf('specPlot: %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "aout_2_specPlot.png", verbose);
    catch ME
        fprintf('    [Error] specPlot: %s\n', ME.message);
    end

    % =========================================================================
    % Tool 3: specPlotPhase - Phase-domain Error
    % =========================================================================
    try
        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        [h, spec, phi, bin] = specPlotPhase(read_data, 'harmonic', 10);
        title(sprintf('specPlotPhase: %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "aout_3_specPlotPhase.png", verbose);
    catch ME
        fprintf('    [Error] specPlotPhase: %s\n', ME.message);
    end

    % =========================================================================
    % Compute error data using sineFit for error-based tools
    % =========================================================================
    try
        [data_fit, freq_est, mag, dc, phi] = sineFit(read_data);
        err_data = read_data - data_fit;
    catch ME
        err_data = read_data - mean(read_data); % Fallback
    end

    % =========================================================================
    % Tool 4: errHistSine (code mode) - Error Histogram by Code
    % =========================================================================
    try
        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        [emean_code, erms_code, code_axis, ~, ~, ~, ~] = errHistSine(read_data, 'bin', 20, 'fin', freqCal, 'disp', 1, 'mode', 1, 'polyorder', 3);
        sgtitle(sprintf('errHistSine (code): %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "aout_4_errHistSine_code.png", verbose);
    catch ME
        fprintf('    [Error] errHistSine (code): %s\n', ME.message);
    end

    % =========================================================================
    % Tool 5: errHistSine (phase mode) - Error Histogram by Phase
    % =========================================================================
    try
        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        [emean, erms, phase_code, anoi, pnoi, ~, ~] = errHistSine(read_data, 'bin', 99, 'fin', freqCal, 'disp', 1, 'mode', 0);
        sgtitle(sprintf('errHistSine (phase): %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "aout_5_errHistSine_phase.png", verbose);
    catch ME
        fprintf('    [Error] errHistSine (phase): %s\n', ME.message);
    end

    % =========================================================================
    % Tool 6: errPDF - Error PDF
    % =========================================================================
    try
        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        [~, mu, sigma, KL_divergence, x, fx, gauss_pdf] = errPDF(err_data, ...
            'Resolution', Resolution, 'FullScale', FullScale);
        title(sprintf('errPDF: %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "aout_6_errPDF.png", verbose);
    catch ME
        fprintf('    [Error] errPDF: %s\n', ME.message);
    end

    % =========================================================================
    % Tool 7: errAutoCorrelation - Error Autocorrelation
    % =========================================================================
    try
        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        [acf, lags] = errAutoCorrelation(err_data, 'MaxLag', 200, 'Normalize', true);
        title(sprintf('errAutoCorrelation: %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "aout_7_errAutoCorrelation.png", verbose);
    catch ME
        fprintf('    [Error] errAutoCorrelation: %s\n', ME.message);
    end

    % =========================================================================
    % Tool 8: Error Spectrum - specPlot on error data
    % =========================================================================
    try
        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        [~, ~, ~, ~, ~, ~, ~, h] = specPlot(err_data, 'label', 0);
        title(sprintf('errSpectrum: %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "aout_8_errSpectrum.png", verbose);
    catch ME
        fprintf('    [Error] errSpectrum: %s\n', ME.message);
    end

    % =========================================================================
    % Tool 9: errEnvelopeSpectrum - Error Envelope Spectrum
    % =========================================================================
    try
        figure('Position', [100, 100, 800, 600], 'Visible', verbose);
        errEnvelopeSpectrum(err_data, 'Fs', 1);
        title(sprintf('errEnvelopeSpectrum: %s', datasetName), 'Interpreter', 'none');
        set(gca, "FontSize", 16);
        saveFig(subFolder, "aout_9_errEnvelopeSpectrum.png", verbose);
    catch ME
        fprintf('    [Error] errEnvelopeSpectrum: %s\n', ME.message);
    end

    %%
    % =========================================================================
    % Create Panel Overview (3x3 grid)
    % =========================================================================
    
    plotFiles = {; ...
        fullfile(subFolder, 'aout_1_tomDecomp.png'); ...
        fullfile(subFolder, 'aout_2_specPlot.png'); ...
        fullfile(subFolder, 'aout_3_specPlotPhase.png'); ...
        fullfile(subFolder, 'aout_4_errHistSine_code.png'); ...
        fullfile(subFolder, 'aout_5_errHistSine_phase.png'); ...
        fullfile(subFolder, 'aout_6_errPDF.png'); ...
        fullfile(subFolder, 'aout_7_errAutoCorrelation.png'); ...
        fullfile(subFolder, 'aout_8_errSpectrum.png'); ...
        fullfile(subFolder, 'aout_9_errEnvelopeSpectrum.png'); ...
        };

    plotLabels = {; ...
        '(1) tomDecomp'; ...
        '(2) specPlot'; ...
        '(3) specPlotPhase'; ...
        '(4) errHistSine (code)'; ...
        '(5) errHistSine (phase)'; ...
        '(6) errPDF'; ...
        '(7) errAutoCorrelation'; ...
        '(8) errSpectrum'; ...
        '(9) errEnvelopeSpectrum'; ...
        };

    fig = figure('Position', [50, 50, 1800, 1600], 'Visible', verbose);
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

    sgtitle(sprintf('AOUT Toolset: %s', datasetName), 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

    panelPath = fullfile(subFolder, 'PANEL_AOUT.png');
    exportgraphics(fig, panelPath, 'Resolution', 600);
    fprintf('  [%s]->[%s]\n', mfilename, panelPath);
end
