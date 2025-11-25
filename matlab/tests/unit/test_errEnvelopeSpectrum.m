%% test_errEnvelopeSpectrum.m - Unit test for errEnvelopeSpectrum function
% Tests the errEnvelopeSpectrum function with sinewave error data
%
% Output structure:
%   test_output/<data_set_name>/test_errEnvelopeSpectrum/
%       errEnvelopeSpectrum_matlab.png

close all; clc; clear;

%% Configuration
addpath('matlab/aout');
addpath('matlab/common');
addpath('matlab/test/unit');

inputDir = "test_data";
outputDir = "test_output";

% Test datasets - leave empty to auto-search
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Test Loop
fprintf('=== test_errEnvelopeSpectrum.m ===\n');
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
    subFolder = fullfile(outputDir, datasetName, 'test_errEnvelopeSpectrum');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Compute error data using sineFit
    [data_fit, ~, ~, ~, ~] = sineFit(read_data);
    err_data = read_data - data_fit;

    %% Run errEnvelopeSpectrum
    figure('Visible', 'off');
    errEnvelopeSpectrum(err_data, 'Fs', 1);
    title(['errEnvelopeSpectrum: ', titleString]);

    % Save plot
    plotPath = fullfile(subFolder, 'errEnvelopeSpectrum_matlab.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n\n', plotPath);
    close(gcf);
end

fprintf('[test_errEnvelopeSpectrum COMPLETE]\n');
