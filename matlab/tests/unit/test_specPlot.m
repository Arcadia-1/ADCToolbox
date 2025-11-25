%% test_specPlot.m - Unit test for specPlot function
% Tests the specPlot function with various sinewave datasets
%
% Output structure:
%   test_output/<data_set_name>/test_specPlot/
%       metrics_matlab.csv      - ENoB, SNDR, SFDR, SNR, THD, pwr, NF
%       spectrum_matlab.png     - Spectrum plot

close all; clc; clear;

%% Configuration
inputDir = "dataset";
outputDir = "test_output";

% Test datasets - leave empty to auto-search
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv', 'batch_sinewave_*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Test Loop
fprintf('=== test_specPlot.m ===\n');
fprintf('[Testing] %d datasets...\n\n', length(filesList));

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
    subFolder = fullfile(outputDir, datasetName, 'test_specPlot');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Run specPlot
    figure('Visible', 'off');
    [ENoB, SNDR, SFDR, SNR, THD, pwr, NF, ~] = specPlot(read_data, 'label', 1, 'harmonic', 5, 'OSR', 1);
    title(['specPlot: ', titleString]);

    % Save plot
    plotPath = fullfile(subFolder, 'spectrum_matlab.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);
    close(gcf);

    % Save metrics to CSV
    metricsTable = table(ENoB, SNDR, SFDR, SNR, THD, pwr, NF, ...
        'VariableNames', {'ENoB', 'SNDR', 'SFDR', 'SNR', 'THD', 'pwr', 'NF'});
    metricsPath = fullfile(subFolder, 'metrics_matlab.csv');
    writetable(metricsTable, metricsPath);
    fprintf('  [Saved] %s\n', metricsPath);

    % Print results
    fprintf('  [Results] ENoB=%.2f, SNDR=%.2f dB, SFDR=%.2f dB, SNR=%.2f dB, THD=%.2f dB\n\n', ...
        ENoB, SNDR, SFDR, SNR, THD);
end

fprintf('[test_specPlot COMPLETE]\n');
