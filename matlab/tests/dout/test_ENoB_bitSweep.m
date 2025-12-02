%% test_ENoB_bitSweep.m - Unit test for ENoB_bitSweep function
close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = fullfile("dataset");
outputDir = "test_data";
figureDir = "test_plots";

filesList ={};
filesList = autoSearchFiles(filesList, inputDir, 'dout_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [ENoB_sweep, nBits_vec] = ENoB_bitSweep(read_data, ...
        'freq', 0, 'order', 5, 'harmonic', 5, 'OSR', 1, 'winType', @hamming);

    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);

    figureName = sprintf("%s_%s_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, ENoB_sweep, verbose);
    saveVariable(subFolder, nBits_vec, verbose);
end
