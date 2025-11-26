%% test_specPlotPhase.m
close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = "dataset";
outputDir = "test_output";
filesList = autoSearchFiles({}, inputDir, 'sinewave_*.csv', 'batch_sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('\n[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [h, spec, phi, bin] = specPlotPhase(read_data, 'harmonic', 10);

    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);

    saveFig(subFolder, "phase.png", verbose);
    saveVariable(subFolder, spec, verbose);
    saveVariable(subFolder, phi, verbose);
    saveVariable(subFolder, bin, verbose);
end
