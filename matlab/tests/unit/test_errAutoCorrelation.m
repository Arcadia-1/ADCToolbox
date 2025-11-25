%% test_errAutoCorrelation.m - Unit test for errAutoCorrelation function
% Tests the errAutoCorrelation function with sinewave error data
%
% Output structure:
%   test_output/<data_set_name>/test_errAutoCorrelation/
%       acf_data_matlab.csv     - lags, acf
%       errACF_matlab.png       - ACF plot

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
fprintf('=== test_errAutoCorrelation.m ===\n');
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
    subFolder = fullfile(outputDir, datasetName, 'test_errAutoCorrelation');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Compute error data using sineFit
    [data_fit, ~, ~, ~, ~] = sineFit(read_data);
    err_data = read_data - data_fit;

    %% Run errAutoCorrelation
    figure('Visible', 'off');
    [acf, lags] = errAutoCorrelation(err_data, 'MaxLag', 200, 'Normalize', true);
    title(['errAutoCorrelation: ', titleString]);

    % Save plot
    plotPath = fullfile(subFolder, 'errACF_matlab.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);
    close(gcf);

    % Save ACF data to CSV
    acfTable = table(lags', acf', ...
        'VariableNames', {'lags', 'acf'});
    acfPath = fullfile(subFolder, 'acf_data_matlab.csv');
    writetable(acfTable, acfPath);
    fprintf('  [Saved] %s\n\n', acfPath);
end

fprintf('[test_errAutoCorrelation COMPLETE]\n');
