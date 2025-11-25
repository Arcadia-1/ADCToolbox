%% test_errSpectrum.m - Unit test for error spectrum analysis
% Tests spectrum analysis of sinewave error data using specPlot
%
% Output structure:
%   test_output/<data_set_name>/test_errSpectrum/
%       errSpectrum_matlab.png      - Error spectrum plot
%       spectrum_data_matlab.csv    - Spectrum data

close all; clc; clear;

%% Configuration
inputDir = "dataset";
outputDir = "test_output";

% Test datasets - leave empty to auto-search
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Test Loop
fprintf('=== test_errSpectrum.m ===\n');
fprintf('Testing errSpectrum function with %d datasets...\n\n', length(filesList));

for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);

    if ~isfile(dataFilePath)
        fprintf('[%d/%d] %s - NOT FOUND, skipping\n\n', k, length(filesList), currentFilename);
        continue;
    end
    fprintf('[%d/%d] %s - found\n', k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    % Extract dataset name
    [~, datasetName, ~] = fileparts(currentFilename);
    titleString = replace(datasetName, '_', '\_');

    % Create output subfolder
    subFolder = fullfile(outputDir, datasetName, 'test_errSpectrum');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Run error spectrum analysis
    figure('Visible', 'off');
    [data_fit, freq_est, mag, dc, phi] = sineFit(read_data);
    err_data = read_data - data_fit;
    [~, ~, ~, ~, ~, ~, ~, h] = specPlot(err_data, "label", 0);
    title(['errSpectrum: ', titleString]);

    % Save plot
    plotPath = fullfile(subFolder, 'errSpectrum_matlab.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);
    close(gcf);
end

fprintf('test_errSpectrum complete.\n');
