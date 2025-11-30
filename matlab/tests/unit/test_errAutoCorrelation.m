%% test_errAutoCorrelation.m - Unit test for errAutoCorrelation function
close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = "dataset/aout";
outputDir = "test_output";
filesList = autoSearchFiles({}, inputDir, 'sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    [~, datasetName, ~] = fileparts(currentFilename);
    titleString = replace(datasetName, '_', '\_');

    [data_fit, ~, ~, ~, ~] = sineFit(read_data);
    err_data = read_data - data_fit;

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [acf, lags] = errAutoCorrelation(err_data, 'MaxLag', 200, 'Normalize', 0);
    title(['errAutoCorrelation: ', titleString]);

    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveFig(subFolder, "errACF_matlab.png", verbose);
    saveVariable(subFolder, lags, verbose);
    saveVariable(subFolder, acf, verbose);
end
