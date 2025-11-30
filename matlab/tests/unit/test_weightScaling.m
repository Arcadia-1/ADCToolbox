%% test_weightScaling.m
close all; clc; clear;
%% Configuration
verbose = 0;
inputDir = "dataset/dout";
outputDir = "test_output";
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'dout_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end
%% Calibration Configuration
Order = 5; % Polynomial order for FGCalSine
%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    bits = readmatrix(dataFilePath);

    [weight_cal, ~, ~, ~, ~, ~] = FGCalSine(bits, 'freq', 0, 'order', Order);

    % Run weightScaling tool
    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    radix = weightScaling(weight_cal);
    set(gca, "FontSize", 16);

    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveFig(subFolder, "weightScaling.png", verbose);
    saveVariable(subFolder, radix, verbose);
    saveVariable(subFolder, weight_cal, verbose);
end
