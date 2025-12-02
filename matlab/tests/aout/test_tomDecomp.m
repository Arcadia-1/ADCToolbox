%% test_tomDecomp.m
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

    read_data = readmatrix(dataFilePath);

    [~, datasetName, ~] = fileparts(currentFilename);
    titleString = replace(datasetName, '_', '\_');

    relative_fin = findFin(read_data);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [signal, error, indep, dep, phi] = tomDecomp(read_data, relative_fin, 10, 1);
    title(['tomDecomp: ', titleString]);
    set(gca, "FontSize",16)

    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveFig(subFolder, "tomDecomp_matlab.png", verbose);
    saveVariable(subFolder, signal, verbose);
    saveVariable(subFolder, error, verbose);
    saveVariable(subFolder, indep, verbose);
    saveVariable(subFolder, dep, verbose);
    saveVariable(subFolder, phi, verbose);
end
