%% test_fft_coefficient_extraction.m - Test FFT-based coefficient extraction
% Extracts transfer function coefficients from harmonic amplitudes in FFT
% This is more accurate than polynomial fitting for sine wave tests

close all; clc; clear; warning("off")

%% Configuration
inputDir = "dataset/non_lin";
outputDir = "test_output";

% Test datasets
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_HD*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Test Loop
fprintf('=== test_fft_coefficient_extraction.m ===\n');
fprintf('[Testing] %d datasets using FFT harmonic extraction\n\n', length(filesList));

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
    subFolder = fullfile(outputDir, datasetName, 'test_fft_extraction');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    % Get expected coefficients
    [k0_exp, k1_exp, k2_exp, k3_exp, k4_exp, k5_exp] = calcExpectedCoeffs(currentFilename);
    fprintf('  [Expected Coefficients from Generation]\n');
    fprintf('    k1 (gain)   = %.6e\n', k1_exp);
    if ~isnan(k2_exp), fprintf('    k2 (HD2)    = %.6e\n', k2_exp); end
    if ~isnan(k3_exp), fprintf('    k3 (HD3)    = %.6e\n', k3_exp); end
    if ~isnan(k4_exp), fprintf('    k4 (HD4)    = %.6e\n', k4_exp); end
    if ~isnan(k5_exp), fprintf('    k5 (HD5)    = %.6e\n', k5_exp); end

    % Get frequency
    [~, freq, ~, ~, ~] = sineFit(read_data);

    % Extract coefficients from FFT
    [k1, k2, k3, k4, k5] = extractCoeffsFromFFT(read_data, freq);

    fprintf('  [Extracted Coefficients (FFT Method)]\n');
    fprintf('    k1 (gain)   = %.6e\n', k1);
    fprintf('    k2 (HD2)    = %.6e\n', k2);
    fprintf('    k3 (HD3)    = %.6e\n', k3);
    fprintf('    k4 (HD4)    = %.6e\n', k4);
    fprintf('    k5 (HD5)    = %.6e\n', k5);

    % Compare
    fprintf('  [Comparison: Extracted vs Expected]\n');
    fprintf('    k1: extracted=%.6e, expected=%.6e, error=%.2f%%\n', ...
        k1, k1_exp, abs(k1 - k1_exp) / abs(k1_exp) * 100);

    if ~isnan(k2_exp) && abs(k2_exp) > 1e-10
        fprintf('    k2: extracted=%.6e, expected=%.6e, error=%.2f%%\n', ...
            k2, k2_exp, abs(k2 - k2_exp) / abs(k2_exp) * 100);
    end

    if ~isnan(k3_exp) && abs(k3_exp) > 1e-10
        fprintf('    k3: extracted=%.6e, expected=%.6e, error=%.2f%%\n', ...
            k3, k3_exp, abs(k3 - k3_exp) / abs(k3_exp) * 100);
    end

    if ~isnan(k4_exp) && abs(k4_exp) > 1e-10
        fprintf('    k4: extracted=%.6e, expected=%.6e, error=%.2f%%\n', ...
            k4, k4_exp, abs(k4 - k4_exp) / abs(k4_exp) * 100);
    end

    if ~isnan(k5_exp) && abs(k5_exp) > 1e-10
        fprintf('    k5: extracted=%.6e, expected=%.6e, error=%.2f%%\n', ...
            k5, k5_exp, abs(k5 - k5_exp) / abs(k5_exp) * 100);
    end

    % Save extracted coefficients
    tfCoeffs = [k1, k2, k3, k4, k5];
    tfNames = {'k1_gain', 'k2_HD2', 'k3_HD3', 'k4_HD4', 'k5_HD5'};
    tfTable = array2table(tfCoeffs, 'VariableNames', tfNames);
    tfPath = fullfile(subFolder, 'coefficients_extracted_fft.csv');
    writetable(tfTable, tfPath);
    fprintf('  [Saved] %s\n', tfPath);

    % Save comparison
    compNames = {'k1_gain'};
    compExpected = [k1_exp];
    compExtracted = [k1];
    compError = [abs(k1 - k1_exp) / abs(k1_exp) * 100];

    if ~isnan(k2_exp) && abs(k2_exp) > 1e-10
        compNames = [compNames, {'k2_HD2'}];
        compExpected = [compExpected, k2_exp];
        compExtracted = [compExtracted, k2];
        compError = [compError, abs(k2 - k2_exp) / abs(k2_exp) * 100];
    end

    if ~isnan(k3_exp) && abs(k3_exp) > 1e-10
        compNames = [compNames, {'k3_HD3'}];
        compExpected = [compExpected, k3_exp];
        compExtracted = [compExtracted, k3];
        compError = [compError, abs(k3 - k3_exp) / abs(k3_exp) * 100];
    end

    if ~isnan(k4_exp) && abs(k4_exp) > 1e-10
        compNames = [compNames, {'k4_HD4'}];
        compExpected = [compExpected, k4_exp];
        compExtracted = [compExtracted, k4];
        compError = [compError, abs(k4 - k4_exp) / abs(k4_exp) * 100];
    end

    if ~isnan(k5_exp) && abs(k5_exp) > 1e-10
        compNames = [compNames, {'k5_HD5'}];
        compExpected = [compExpected, k5_exp];
        compExtracted = [compExtracted, k5];
        compError = [compError, abs(k5 - k5_exp) / abs(k5_exp) * 100];
    end

    compTable = table(compNames', compExpected', compExtracted', compError', ...
        'VariableNames', {'Coefficient', 'Expected', 'Extracted', 'Error_Percent'});
    compPath = fullfile(subFolder, 'comparison_fft.csv');
    writetable(compTable, compPath);
    fprintf('  [Saved] %s\n', compPath);

    fprintf('\n');
end

fprintf('[test_fft_coefficient_extraction COMPLETE]\n');
