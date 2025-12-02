%% test_tomDecomp.m
close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = fullfile("dataset");
outputDir = "test_output";
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
    titleString = replace(datasetName, '_', '\_');

    relative_fin = findfreq(read_data);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [signal, error, indep, dep, phi] = tomdec(read_data, relative_fin, 10, 1);
    title(['tomdec: ', titleString]);
    set(gca, "FontSize",16)

    subFolder = fullfile(outputDir, datasetName, mfilename);

    figureName = sprintf("%s_%s_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, signal, verbose);
    saveVariable(subFolder, error, verbose);
    saveVariable(subFolder, indep, verbose);
    saveVariable(subFolder, dep, verbose);
    saveVariable(subFolder, phi, verbose);
end
