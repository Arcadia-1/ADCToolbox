%% test_static_nonlinearity.m - Test polynomial extraction for STATIC nonlinearity
% Tests polynomial fitting to extract static transfer function coefficients
%
% Output structure:
%   test_output/<data_set_name>/test_static_nonlinearity/
%       polycoeff_matlab.csv            - polynomial coefficients
%       comparison_matlab.csv            - extracted vs expected
%       static_transfer_function.png    - plot

close all; clc; clear; warning("off")

%% Configuration
inputDir = "dataset/static_nonlin";  % Subfolder for static nonlinearity data
outputDir = "test_output";

% Polynomial order to test (should match generation order)
polyOrder = 3;  % Will extract k1, k2, k3

% Test datasets - leave empty to auto-search
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'static_nonlin*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Storage for summary
summaryNames = {};
summaryK1_exp = [];
summaryK1_ext = [];
summaryK2_exp = [];
summaryK2_ext = [];
summaryK3_exp = [];
summaryK3_ext = [];

%% Test Loop
fprintf('=== test_static_nonlinearity.m ===\n');
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
    titleString = replace(datasetName, '_', '\_');

    % Create output subfolder
    subFolder = fullfile(outputDir, datasetName, 'test_static_nonlinearity');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    % Get expected coefficients from filename
    [k1_exp, k2_exp, k3_exp, k4_exp, k5_exp] = calcExpectedCoeffsStatic(currentFilename);
    fprintf('  [Expected Coefficients from Generation]\n');
    fprintf('    k1 (gain) = %.6e\n', k1_exp);
    if ~isnan(k2_exp), fprintf('    k2        = %.6e\n', k2_exp); end
    if ~isnan(k3_exp), fprintf('    k3        = %.6e\n', k3_exp); end
    if ~isnan(k4_exp), fprintf('    k4        = %.6e\n', k4_exp); end
    if ~isnan(k5_exp), fprintf('    k5        = %.6e\n', k5_exp); end

    % Get frequency
    [~, freq, ~, ~, ~] = sineFit(read_data);

    % Extract transfer function using direct polynomial fit
    [k0_fit, k1_fit, k2_fit, k3_fit, k4_fit, k5_fit, x_ideal, y_actual, polycoeff] = ...
        extractTransferFunction(read_data, freq, polyOrder);

    fprintf('  [Extracted Coefficients (Polynomial Fit)]\n');
    fprintf('    k0 (offset) = %.6e  [should be ~0]\n', k0_fit);
    fprintf('    k1 (gain)   = %.6e\n', k1_fit);
    fprintf('    k2          = %.6e\n', k2_fit);
    fprintf('    k3          = %.6e\n', k3_fit);
    if polyOrder >= 4, fprintf('    k4          = %.6e\n', k4_fit); end
    if polyOrder >= 5, fprintf('    k5          = %.6e\n', k5_fit); end

    % Compare extracted vs expected
    fprintf('  [Comparison: Extracted vs Expected]\n');
    fprintf('    k1: extracted=%.6e, expected=%.6e, error=%.2f%%\n', ...
        k1_fit, k1_exp, abs(k1_fit - k1_exp) / abs(k1_exp) * 100);

    if ~isnan(k2_exp) && abs(k2_exp) > 1e-10
        fprintf('    k2: extracted=%.6e, expected=%.6e, error=%.2f%%\n', ...
            k2_fit, k2_exp, abs(k2_fit - k2_exp) / abs(k2_exp) * 100);
    end

    if ~isnan(k3_exp) && abs(k3_exp) > 1e-10
        fprintf('    k3: extracted=%.6e, expected=%.6e, error=%.2f%%\n', ...
            k3_fit, k3_exp, abs(k3_fit - k3_exp) / abs(k3_exp) * 100);
    end

    if polyOrder >= 4 && ~isnan(k4_exp) && abs(k4_exp) > 1e-10
        fprintf('    k4: extracted=%.6e, expected=%.6e, error=%.2f%%\n', ...
            k4_fit, k4_exp, abs(k4_fit - k4_exp) / abs(k4_exp) * 100);
    end

    if polyOrder >= 5 && ~isnan(k5_exp) && abs(k5_exp) > 1e-10
        fprintf('    k5: extracted=%.6e, expected=%.6e, error=%.2f%%\n', ...
            k5_fit, k5_exp, abs(k5_fit - k5_exp) / abs(k5_exp) * 100);
    end

    % Save extracted coefficients
    tfCoeffs = [k0_fit, k1_fit, k2_fit, k3_fit];
    tfNames = {'k0_offset', 'k1_gain', 'k2', 'k3'};
    if polyOrder >= 4
        tfCoeffs = [tfCoeffs, k4_fit];
        tfNames = [tfNames, {'k4'}];
    end
    if polyOrder >= 5
        tfCoeffs = [tfCoeffs, k5_fit];
        tfNames = [tfNames, {'k5'}];
    end
    tfTable = array2table(tfCoeffs, 'VariableNames', tfNames);
    tfPath = fullfile(subFolder, 'coefficients_extracted.csv');
    writetable(tfTable, tfPath);
    fprintf('  [Saved] %s\n', tfPath);

    % Save expected coefficients
    expCoeffs = [k1_exp];
    expNames = {'k1_gain_expected'};
    if ~isnan(k2_exp)
        expCoeffs = [expCoeffs, k2_exp];
        expNames = [expNames, {'k2_expected'}];
    end
    if ~isnan(k3_exp)
        expCoeffs = [expCoeffs, k3_exp];
        expNames = [expNames, {'k3_expected'}];
    end
    if ~isnan(k4_exp)
        expCoeffs = [expCoeffs, k4_exp];
        expNames = [expNames, {'k4_expected'}];
    end
    if ~isnan(k5_exp)
        expCoeffs = [expCoeffs, k5_exp];
        expNames = [expNames, {'k5_expected'}];
    end
    expTable = array2table(expCoeffs, 'VariableNames', expNames);
    expPath = fullfile(subFolder, 'coefficients_expected.csv');
    writetable(expTable, expPath);
    fprintf('  [Saved] %s\n', expPath);

    % Save comparison
    compNames = {'k1_gain'};
    compExpected = [k1_exp];
    compExtracted = [k1_fit];
    compError = [abs(k1_fit - k1_exp) / abs(k1_exp) * 100];

    if ~isnan(k2_exp) && abs(k2_exp) > 1e-10
        compNames = [compNames, {'k2'}];
        compExpected = [compExpected, k2_exp];
        compExtracted = [compExtracted, k2_fit];
        compError = [compError, abs(k2_fit - k2_exp) / abs(k2_exp) * 100];
    end

    if ~isnan(k3_exp) && abs(k3_exp) > 1e-10
        compNames = [compNames, {'k3'}];
        compExpected = [compExpected, k3_exp];
        compExtracted = [compExtracted, k3_fit];
        compError = [compError, abs(k3_fit - k3_exp) / abs(k3_exp) * 100];
    end

    if polyOrder >= 4 && ~isnan(k4_exp) && abs(k4_exp) > 1e-10
        compNames = [compNames, {'k4'}];
        compExpected = [compExpected, k4_exp];
        compExtracted = [compExtracted, k4_fit];
        compError = [compError, abs(k4_fit - k4_exp) / abs(k4_exp) * 100];
    end

    if polyOrder >= 5 && ~isnan(k5_exp) && abs(k5_exp) > 1e-10
        compNames = [compNames, {'k5'}];
        compExpected = [compExpected, k5_exp];
        compExtracted = [compExtracted, k5_fit];
        compError = [compError, abs(k5_fit - k5_exp) / abs(k5_exp) * 100];
    end

    compTable = table(compNames', compExpected', compExtracted', compError', ...
        'VariableNames', {'Coefficient', 'Expected', 'Extracted', 'Error_Percent'});
    compPath = fullfile(subFolder, 'comparison.csv');
    writetable(compTable, compPath);
    fprintf('  [Saved] %s\n', compPath);

    % Plot static transfer function
    figure('Position', [100, 100, 1200, 400], 'Visible', 'on');

    % Sort data by x for proper transfer curve visualization
    [x_sorted, sort_idx] = sort(x_ideal);
    y_sorted = y_actual(sort_idx);

    subplot(1,3,1);
    plot(x_sorted, y_sorted, 'b-', 'LineWidth', 1);
    hold on;
    x_plot = linspace(min(x_ideal), max(x_ideal), 500);
    x_max = max(abs(x_ideal));
    x_plot_norm = x_plot / x_max;
    y_plot = polyval(polycoeff, x_plot_norm);
    plot(x_plot, y_plot, 'r--', 'LineWidth', 2);
    xlabel('Input x');
    ylabel('Output y (DC removed)');
    title('Static Transfer Curve');
    legend('Data (sorted)', sprintf('Fit: y = %.2e + %.3fx + %.2ex^2 + %.2ex^3', k0_fit, k1_fit, k2_fit, k3_fit), ...
        'Location', 'northwest');
    grid on;

    subplot(1,3,2);
    % Plot INL (deviation from linear)
    y_linear = k1_fit * x_sorted;  % Linear component
    inl = y_sorted - y_linear;
    plot(x_sorted, inl, 'b-', 'LineWidth', 1);
    hold on;
    inl_fit = k0_fit + k2_fit * x_plot.^2 + k3_fit * x_plot.^3;
    plot(x_plot, inl_fit, 'r--', 'LineWidth', 2);
    xlabel('Input x');
    ylabel('INL (y - k1*x)');
    title('Integral Nonlinearity (INL)');
    legend('Data', 'Fit (k0 + k2*x^2 + k3*x^3)');
    grid on;

    subplot(1,3,3);
    residuals = y_sorted - polyval(polycoeff, x_sorted / x_max);
    plot(x_sorted, residuals, 'r-', 'LineWidth', 1);
    xlabel('Input x');
    ylabel('Residual (y - fit)');
    title(sprintf('Fit Residuals\nRMS: %.3e', rms(residuals)));
    grid on;
    yline(0, 'k--', 'LineWidth', 1);

    sgtitle(titleString);

    plotPath = fullfile(subFolder, 'static_transfer_function.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);

    % Don't close figure - let user see it
    % close(gcf);

    % Store for summary
    summaryNames{end+1} = datasetName;
    summaryK1_exp(end+1) = k1_exp;
    summaryK1_ext(end+1) = k1_fit;
    summaryK2_exp(end+1) = k2_exp;
    summaryK2_ext(end+1) = k2_fit;
    summaryK3_exp(end+1) = k3_exp;
    summaryK3_ext(end+1) = k3_fit;

    fprintf('\n');
end

%% Create Summary Comparison Plot
if ~isempty(summaryNames)
    figure('Position', [100, 100, 1200, 800], 'Visible', 'on');

    % k1 comparison
    subplot(3,2,1);
    x_pos = 1:length(summaryNames);
    bar(x_pos, [summaryK1_exp; summaryK1_ext]');
    set(gca, 'XTickLabel', {});
    ylabel('k1 (gain)');
    title('k1: Expected vs Extracted');
    legend('Expected', 'Extracted', 'Location', 'best');
    grid on;

    subplot(3,2,2);
    error_k1 = abs(summaryK1_ext - summaryK1_exp) ./ abs(summaryK1_exp) * 100;
    bar(x_pos, error_k1);
    set(gca, 'XTickLabel', {});
    ylabel('Error (%)');
    title('k1 Error');
    grid on;

    % k2 comparison
    subplot(3,2,3);
    bar(x_pos, [summaryK2_exp; summaryK2_ext]');
    set(gca, 'XTickLabel', {});
    ylabel('k2');
    title('k2: Expected vs Extracted');
    legend('Expected', 'Extracted', 'Location', 'best');
    grid on;

    subplot(3,2,4);
    error_k2 = abs(summaryK2_ext - summaryK2_exp) ./ abs(summaryK2_exp) * 100;
    bar(x_pos, error_k2);
    set(gca, 'XTickLabel', {});
    ylabel('Error (%)');
    title('k2 Error');
    grid on;

    % k3 comparison
    subplot(3,2,5);
    bar(x_pos, [summaryK3_exp; summaryK3_ext]');
    set(gca, 'XTick', x_pos);
    set(gca, 'XTickLabel', summaryNames, 'XTickLabelRotation', 45);
    ylabel('k3');
    title('k3: Expected vs Extracted');
    legend('Expected', 'Extracted', 'Location', 'best');
    grid on;

    subplot(3,2,6);
    error_k3 = abs(summaryK3_ext - summaryK3_exp) ./ abs(summaryK3_exp) * 100;
    bar(x_pos, error_k3);
    set(gca, 'XTick', x_pos);
    set(gca, 'XTickLabel', summaryNames, 'XTickLabelRotation', 45);
    ylabel('Error (%)');
    title('k3 Error');
    grid on;

    sgtitle('Static Nonlinearity Extraction - Summary Comparison');

    % Save summary plot
    summaryPath = fullfile(outputDir, 'static_nonlinearity_summary.png');
    saveas(gcf, summaryPath);
    fprintf('\n[Saved Summary] %s\n', summaryPath);
end

fprintf('[test_static_nonlinearity COMPLETE]\n');
fprintf('Figures are displayed for review. Close them to continue.\n');
