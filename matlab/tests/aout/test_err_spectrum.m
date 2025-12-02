close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = "reference_dataset/sinewave";
outputDir = "reference_output";
figureDir = "test_plots";

filesList ={};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);
    [~, datasetName, ~] = fileparts(currentFilename);

    % Compute error data using sineFit
    [data_fit, ~, ~, ~, ~] = sineFit(read_data);
    err_data = read_data - data_fit;

    % Run specPlot on error data
    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [ENoB, SNDR, SFDR, SNR, THD, pwr, NF, ~, ~] = specPlot(err_data, 'label', 0);

    title(mfilename);

    % Save outputs
    subFolder = fullfile(outputDir, datasetName, mfilename);
    figureName = sprintf("%s_%s_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, ENoB, verbose);
    saveVariable(subFolder, SNDR, verbose);
    saveVariable(subFolder, SFDR, verbose);
    saveVariable(subFolder, SNR, verbose);
    saveVariable(subFolder, THD, verbose);
    saveVariable(subFolder, pwr, verbose);
    saveVariable(subFolder, NF, verbose);
end
