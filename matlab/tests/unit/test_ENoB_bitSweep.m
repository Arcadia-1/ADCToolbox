%% test_ENoB_bitSweep.m - Unit test for ENoB_bitSweep function
% Tests the ENoB_bitSweep function with digital ADC output data
%
% Output structure:
%   test_output/<data_set_name>/test_ENoB_bitSweep/
%       ENoB_sweep_matlab.png       - ENoB sweep plot
%       ENoB_sweep_data_matlab.csv  - nBits, ENoB data

close all; clc; clear;

%% Configuration
inputDir = "dataset";
outputDir = "test_output";

% Test datasets - leave empty to auto-search
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'dout_*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Test Loop
fprintf('=== test_ENoB_bitSweep.m ===\n');
fprintf('[Testing] %d datasets...\n\n', length(filesList));

for k = 1:length(filesList)*0+1
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
    subFolder = fullfile(outputDir, datasetName, 'test_ENoB_bitSweep');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Run ENoB_bitSweep
    figure('Visible', 'off');
    [ENoB_sweep, nBits_vec] = ENoB_bitSweep(read_data, 'freq', 0, 'order', 5, ...
        'harmonic', 5, 'OSR', 1, 'winType', @hamming, 'plot', 1);
    title(['ENoB Bit Sweep: ', titleString]);

    % Save plot
    plotPath = fullfile(subFolder, 'ENoB_sweep_matlab.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);
    % close(gcf);

    % Save data to CSV
    sweepTable = table(nBits_vec', ENoB_sweep', ...
        'VariableNames', {'nBits', 'ENoB'});
    dataPath = fullfile(subFolder, 'ENoB_sweep_data_matlab.csv');
    writetable(sweepTable, dataPath);
    fprintf('  [Saved] %s\n', dataPath);

    % Print summary
    maxENoB = max(ENoB_sweep(~isnan(ENoB_sweep)));
    maxIdx = find(ENoB_sweep == maxENoB, 1);
    fprintf('  [Results] Max ENoB = %.2f bits (using %d bits)\n\n', maxENoB, maxIdx);
end

fprintf('[test_ENoB_bitSweep COMPLETE]\n');
