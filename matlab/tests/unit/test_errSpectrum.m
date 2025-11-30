%% test_errSpectrum.m - Unit test for error spectrum analysis
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

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [data_fit, freq_est, mag, dc, phi] = sineFit(read_data);
    err_data = read_data - data_fit;
    specPlot(err_data, "label", 0);
    title(['errSpectrum: ', titleString]);
    set(gca, "FontSize", 16);

    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveFig(subFolder, "errSpectrum_matlab.png", verbose);
end
