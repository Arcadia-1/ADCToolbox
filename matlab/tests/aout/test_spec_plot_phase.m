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

    % Run plotphase (previously specPlotPhase)
    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    h = plotphase(read_data);

    % Save figure
    subFolder = fullfile(outputDir, datasetName, mfilename);
    figureName = sprintf("%s_phase_matlab.png", datasetName);
    saveFig(figureDir, figureName, verbose);
end
