%% test_ENoB_bitSweep.m - Unit test for ENoB_bitSweep function
% Output: test_output/<dataset>/test_ENoB_bitSweep/
%   - ENoB_sweep_matlab.png
%   - ENoB_sweep_data_matlab.csv

close all; clc; clear;

%% Configuration
inputDir = "dataset";
outputDir = "test_output";
filesList = autoSearchFiles({}, inputDir, 'dout_*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Test Loop
fprintf('=== test_ENoB_bitSweep.m ===\n');
fprintf('[Testing] %d datasets...\n\n', length(filesList));

for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);

    if ~isfile(dataFilePath)
        fprintf('[%d/%d] %s - NOT FOUND\n\n', k, length(filesList), currentFilename);
        continue;
    end
    fprintf('[%d/%d] [Processing] %s\n', k, length(filesList), currentFilename);

    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, 'test_ENoB_bitSweep');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Run ENoB_bitSweep
    figure('Position', [100, 100, 800, 600], 'Visible', 'off');
    [ENoB_sweep, nBits_vec] = ENoB_bitSweep(readmatrix(dataFilePath), ...
        'freq', 0, 'order', 5, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);

    title(replace(datasetName, '_', '\_'));

    % Save results
    saveas(gcf, fullfile(subFolder, 'ENoB_sweep_matlab.png'));
    fprintf('  [Saved] ENoB_sweep_matlab.png\n');
    % close(gcf);

    writetable(table(nBits_vec', ENoB_sweep', 'VariableNames', {'nBits', 'ENoB'}), ...
        fullfile(subFolder, 'ENoB_sweep_data_matlab.csv'));
    fprintf('  [Saved] ENoB_sweep_data_matlab.csv\n');

    % Summary
    maxENoB = max(ENoB_sweep(~isnan(ENoB_sweep)));
    fprintf('  [Results] Max ENoB = %.2f bits (using %d bits)\n\n', ...
        maxENoB, find(ENoB_sweep == maxENoB, 1));
end

fprintf('[test_ENoB_bitSweep COMPLETE]\n');
