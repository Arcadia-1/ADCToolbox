%% test_bitActivity.m
close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = "dataset/dout";
outputDir = "test_output";
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'dout_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    bits = readmatrix(dataFilePath);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    bit_usage = bitActivity(bits, 'AnnotateExtremes', true);
    set(gca, "FontSize", 16);

    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveFig(subFolder, "bitActivity_matlab.png", verbose);
    saveVariable(subFolder, bit_usage, verbose);
end
