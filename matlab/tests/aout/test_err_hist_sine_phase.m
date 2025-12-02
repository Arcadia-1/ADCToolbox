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

    % Find fundamental frequency
    freqCal = findFin(read_data);

    % Run errHistSine in phase mode (mode=0)
    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [emean, erms, phase_code, anoi, pnoi, ~, ~, ~, ~, ~, ~] = errHistSine(read_data, 'bin', 360, 'fin', freqCal, 'disp', 1, 'mode', 0);

    sgtitle(sprintf('%s: %s', mfilename, datasetName), 'FontWeight', 'bold');

    % Save outputs
    subFolder = fullfile(outputDir, datasetName, mfilename);
    figureName = sprintf("%s_%s_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, anoi, verbose);
    saveVariable(subFolder, pnoi, verbose);
    saveVariable(subFolder, phase_code, verbose);
    saveVariable(subFolder, emean, verbose);
    saveVariable(subFolder, erms, verbose);
end
