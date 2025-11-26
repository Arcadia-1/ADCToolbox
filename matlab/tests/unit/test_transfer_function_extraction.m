%% test_transfer_function_extraction.m - Test direct transfer function extraction
% Tests extractTransferFunction to get actual transfer function coefficients
%
% This extracts: y_actual = k0 + k1*x + k2*x^2 + k3*x^3 + ...
% where x is the ideal sine wave input

close all; clc; clear; warning("off")

%% Configuration
inputDir = "dataset/non_lin";
outputDir = "test_output";

% Polynomial order
polyOrder = 3;

% Test datasets
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_HD*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Test Loop
fprintf('=== test_transfer_function_extraction.m ===\n');
fprintf('[Testing] %d datasets with polynomial order: %d\n\n', ...
    length(filesList), polyOrder);

for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);

    if ~isfile(dataFilePath)
        fprintf('[%d/%d] %s - NOT FOUND, skipping\n\n', k, length(filesList), currentFilename);
        continue;
    end
    fprintf('[%d/%d] [Processing] %s\n', k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    % Extract dataset name
    [~, datasetName, ~] = fileparts(currentFilename);

    % Create output subfolder
    subFolder = fullfile(outputDir, datasetName, 'test_transfer_function');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    % Get expected coefficients
    [k0_exp, k1_exp, k2_exp, k3_exp, k4_exp, k5_exp] = calcExpectedCoeffs(currentFilename);
    fprintf('  [Expected Coefficients from Generation]\n');
    fprintf('    k0 (offset) = %.6e  [should be ~0 after DC removal]\n', k0_exp);
    fprintf('    k1 (gain)   = %.6e\n', k1_exp);
    if ~isnan(k2_exp), fprintf('    k2 (HD2)    = %.6e\n', k2_exp); end
    if ~isnan(k3_exp), fprintf('    k3 (HD3)    = %.6e\n', k3_exp); end

    % Get frequency
    [~, freq, ~, ~, ~] = sineFit(read_data);

    % Extract transfer function
    [k0, k1, k2, k3, k4, k5, x_ideal, y_actual, polycoeff] = ...
        extractTransferFunction(read_data, freq, polyOrder);

    fprintf('  [Extracted Transfer Function Coefficients]\n');
    fprintf('    k0 (offset) = %.6e\n', k0);
    fprintf('    k1 (gain)   = %.6e\n', k1);
    fprintf('    k2 (HD2)    = %.6e\n', k2);
    fprintf('    k3 (HD3)    = %.6e\n', k3);

    % Compare
    fprintf('  [Comparison: Extracted vs Expected]\n');
    fprintf('    k0: extracted=%.6e, expected=%.6e (expect ~0)\n', k0, 0);
    fprintf('    k1: extracted=%.6e, expected=%.6e, error=%.2f%%\n', ...
        k1, k1_exp, abs(k1 - k1_exp) / abs(k1_exp) * 100);
    if ~isnan(k2_exp)
        fprintf('    k2: extracted=%.6e, expected=%.6e, error=%.2f%%\n', ...
            k2, k2_exp, abs(k2 - k2_exp) / abs(k2_exp) * 100);
    end
    if ~isnan(k3_exp)
        fprintf('    k3: extracted=%.6e, expected=%.6e, error=%.2f%%\n', ...
            k3, k3_exp, abs(k3 - k3_exp) / abs(k3_exp) * 100);
    end

    % Save results
    tfCoeffs = [k0, k1, k2, k3];
    tfNames = {'k0_offset', 'k1_gain', 'k2_HD2', 'k3_HD3'};
    tfTable = array2table(tfCoeffs, 'VariableNames', tfNames);
    tfPath = fullfile(subFolder, 'transfer_function_extracted.csv');
    writetable(tfTable, tfPath);
    fprintf('  [Saved] %s\n', tfPath);

    % Save comparison
    compNames = {'k0_offset', 'k1_gain'};
    compExpected = [0, k1_exp];  % k0 should be 0 after DC removal
    compExtracted = [k0, k1];
    compError = [abs(k0), abs(k1 - k1_exp) / abs(k1_exp) * 100];

    if ~isnan(k2_exp)
        compNames = [compNames, {'k2_HD2'}];
        compExpected = [compExpected, k2_exp];
        compExtracted = [compExtracted, k2];
        compError = [compError, abs(k2 - k2_exp) / abs(k2_exp) * 100];
    end

    if ~isnan(k3_exp)
        compNames = [compNames, {'k3_HD3'}];
        compExpected = [compExpected, k3_exp];
        compExtracted = [compExtracted, k3];
        compError = [compError, abs(k3 - k3_exp) / abs(k3_exp) * 100];
    end

    compTable = table(compNames', compExpected', compExtracted', compError', ...
        'VariableNames', {'Coefficient', 'Expected', 'Extracted', 'Error_Percent'});
    compPath = fullfile(subFolder, 'comparison.csv');
    writetable(compTable, compPath);
    fprintf('  [Saved] %s\n', compPath);

    % Plot transfer function
    figure('Position', [100, 100, 1000, 400], 'Visible', 'off');

    subplot(1,2,1);
    plot(x_ideal, y_actual, 'b.', 'MarkerSize', 2);
    hold on;
    x_plot = linspace(min(x_ideal), max(x_ideal), 500);
    x_max = max(abs(x_ideal));
    x_plot_norm = x_plot / x_max;
    y_plot = polyval(polycoeff, x_plot_norm);
    plot(x_plot, y_plot, 'r-', 'LineWidth', 2);
    xlabel('Ideal Input (x)');
    ylabel('Actual Output (y, DC removed)');
    title('Transfer Function: y = f(x)');
    legend('Data', sprintf('Fit: y = %.3e + %.3fx + %.3ex^2 + %.3ex^3', k0, k1, k2, k3));
    grid on;

    subplot(1,2,2);
    residuals = y_actual - polyval(polycoeff, x_ideal / x_max);
    plot(x_ideal, residuals, 'r.', 'MarkerSize', 2);
    xlabel('Ideal Input (x)');
    ylabel('Residual (y - fit)');
    title(sprintf('Fit Residuals (RMS: %.6f)', rms(residuals)));
    grid on;

    sgtitle(replace(datasetName, '_', '\_'));

    plotPath = fullfile(subFolder, 'transfer_function_plot.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);
    close(gcf);

    fprintf('\n');
end

fprintf('[test_transfer_function_extraction COMPLETE]\n');
