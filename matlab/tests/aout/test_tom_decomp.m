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
    re_fin = findFin(read_data);

    % Run tomDecomp
    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [signal, error, indep, dep, phi] = tomDecomp(read_data, re_fin, 10, 1);

    sgtitle(sprintf('tomDecomp: %s', datasetName), 'FontWeight', 'bold');

    % Calculate RMS metrics
    rms_error = sqrt(mean(error.^2));
    rms_indep = sqrt(mean(indep.^2));
    rms_dep = sqrt(mean(dep.^2));

    % Save outputs
    subFolder = fullfile(outputDir, datasetName, mfilename);
    figureName = sprintf("%s_%s_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, signal, verbose);
    saveVariable(subFolder, error, verbose);
    saveVariable(subFolder, indep, verbose);
    saveVariable(subFolder, dep, verbose);
    saveVariable(subFolder, phi, verbose);
    saveVariable(subFolder, rms_error, verbose);
    saveVariable(subFolder, rms_indep, verbose);
    saveVariable(subFolder, rms_dep, verbose);
end
