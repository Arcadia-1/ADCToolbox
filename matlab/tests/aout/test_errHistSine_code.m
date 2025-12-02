%% test_errHistSine_code.m - Unit test for errHistSine code mode
close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = "dataset/aout/sinewave";
outputDir = "test_output";

filesList ={};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);

    read_data = readmatrix(dataFilePath);
    [~, freq, ~, ~, ~] = sineFit(read_data);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [emean_code, erms_code, code_axis, ~, ~, ~, ~, ~, k1, k2, k3] = ...
        errHistSine(read_data, 'bin', 256, 'fin', freq, 'disp', 1, 'mode', 1, 'polyorder', 3);
    sgtitle(replace(datasetName, '_', '\_'), 'Interpreter', 'tex');
    set(findall(gcf, 'Type', 'axes'), 'FontSize', 14);

    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveFig(subFolder, "errHistSine_code_matlab.png", verbose);
    saveVariable(subFolder, code_axis, verbose);
    saveVariable(subFolder, emean_code, verbose);
    saveVariable(subFolder, erms_code, verbose);
    saveVariable(subFolder, k1, verbose);
    saveVariable(subFolder, k2, verbose);
    saveVariable(subFolder, k3, verbose);
end
