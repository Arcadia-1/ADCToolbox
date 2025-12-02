%% test_errHistSine_phase.m - Unit test for errHistSine phase mode
close all; clc; clear; warning("off")

%% Configuration
verbose = 0;
inputDir = fullfile("dataset", "sinewave");
outputDir = "test_data";
figureDir = "test_plots";

filesList ={};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv', 'batch_sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);

    read_data = readmatrix(dataFilePath);
    [~, freq, ~, ~, ~] = sinfit(read_data);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [emean, erms, xx, anoi, pnoi] = errsin(read_data, 'bin', 360, 'fin', freq, 'disp', 1, 'xaxis', 'phase');
    sgtitle(replace(datasetName, '_', '\_'), 'Interpreter', 'tex');
    set(findall(gcf, 'Type', 'axes'), 'FontSize', 14);

    subFolder = fullfile(outputDir, datasetName, mfilename);
    figureName = sprintf("%s_%s_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, anoi, verbose);
    saveVariable(subFolder, pnoi, verbose);
    saveVariable(subFolder, xx, verbose);
    saveVariable(subFolder, emean, verbose);
    saveVariable(subFolder, erms, verbose);
end
