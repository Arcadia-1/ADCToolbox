%% test_sineFit.m - Unit test for sineFit function
% Output: test_output/<dataset>/test_sineFit/{metrics,fit_data}_matlab.csv

close all; clc; clear;
addpath('matlab/aout', 'matlab/common', 'matlab/test/unit');

inputDir = "test_data";
outputDir = "test_output";

% Test datasets - leave empty to auto-search
filesList = {
    % "batch_sinewave_Nrun_2.csv"
};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv');

if isempty(filesList)
    error('No test files found in %s', inputDir);
end

%% Test Loop
fprintf('=== test_sineFit.m ===\n');
fprintf('[Testing] %d datasets...\n\n', length(filesList));

for k = 1:length(filesList)
    currentFilename = filesList{k};
    filepath = fullfile(inputDir, currentFilename);

    if ~isfile(filepath)
        fprintf('[%d/%d] %s - NOT FOUND, skipping\n\n', k, length(filesList), currentFilename);
        continue;
    end
    fprintf('[%d/%d] [Processing] %s\n', k, length(filesList), currentFilename);

    % Read and fit
    read_data = readmatrix(filepath);
    [data_fit, freq, mag, dc, phi] = sineFit(read_data);
    n_cols = length(freq);

    % Create output folder
    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, 'test_sineFit');
    if ~isfolder(subFolder), mkdir(subFolder); end

    % Save metrics
    writetable(table(freq(:), mag(:), dc(:), phi(:), ...
        'VariableNames', {'freq','mag','dc','phi'}), ...
        fullfile(subFolder, 'metrics_matlab.csv'));

    % Save fit data (first 1000 samples)
    N_save = min(1000, size(data_fit, 1));
    fitTable = array2table(data_fit(1:N_save, :));
    fitTable.Properties.VariableNames = arrayfun(@(i) sprintf('data_fit_%d', i-1), ...
        1:size(data_fit,2), 'UniformOutput', false);
    writetable(fitTable, fullfile(subFolder, 'fit_data_matlab.csv'));

    % Print summary
    if n_cols == 1
        fprintf('  [Results] size(data_fit)=[%d, %d], freq=%.8f, mag=%.6f, dc=%.6f, phi=%.6f\n\n', size(data_fit), freq, mag, dc, phi);
    else
        fprintf('  [Results] %d cols: freq=%.8f±%.2e, mag=%.6f±%.2e\n\n', ...
            n_cols, mean(freq), std(freq), mean(mag), std(mag));
    end
end

fprintf('[test_sineFit COMPLETE]\n');
