%% test_errHistSine_code.m - Unit test for errHistSine code mode
close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = fullfile("dataset", "sinewave");
outputDir = "test_data";
figureDir = "test_plots";

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
    [~, freq, ~, ~, ~] = sinfit(read_data);

    % Error histogram analysis
    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [emean_code, erms_code, code_axis] = ...
        errsin(read_data, 'bin', 256, 'fin', freq, 'disp', 1, 'xaxis', 'value');
    sgtitle(replace(datasetName, '_', '\_'), 'Interpreter', 'tex');
    set(findall(gcf, 'Type', 'axes'), 'FontSize', 14);

    % Static nonlinearity extraction
    [k1, k2, k3, polycoeff] = fitstaticnl(read_data, 3, freq);
    fprintf('  Static NL: k1=%.6f, k2=%.6f, k3=%.6f\n', k1, k2, k3);

    subFolder = fullfile(outputDir, datasetName, mfilename);

    figureName = sprintf("%s_%s_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, code_axis, verbose);
    saveVariable(subFolder, emean_code, verbose);
    saveVariable(subFolder, erms_code, verbose);
    saveVariable(subFolder, k1, verbose);
    saveVariable(subFolder, k2, verbose);
    saveVariable(subFolder, k3, verbose);
    saveVariable(subFolder, polycoeff, verbose);
end
