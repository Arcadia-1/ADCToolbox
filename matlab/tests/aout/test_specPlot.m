%% test_specPlot.m
close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = "dataset/aout/sinewave";
outputDir = "test_output";

filesList ={};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv', 'batch_sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [ENoB, SNDR, SFDR, SNR, THD, pwr, NF, ~] = specPlot(read_data, 'label', 1, 'harmonic', 5, 'OSR', 1);
    set(gca, "FontSize",16)

    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveFig(subFolder, "specPlot_matlab.png", verbose);
    saveVariable(subFolder, ENoB, verbose);
    saveVariable(subFolder, SNDR, verbose);
    saveVariable(subFolder, SFDR, verbose);
    saveVariable(subFolder, SNR, verbose);
    saveVariable(subFolder, THD, verbose);
    saveVariable(subFolder, pwr, verbose);
    saveVariable(subFolder, NF, verbose);
end
