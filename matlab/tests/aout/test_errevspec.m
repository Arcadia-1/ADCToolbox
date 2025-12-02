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

    [data_fit, ~, ~, ~, ~] = sinfit(read_data);
    err_data = read_data - data_fit;

    e = err_data(:);
    env = abs(hilbert(e));

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [ENoB, SNDR, SFDR, SNR, THD, pwr, NF, ~] = errEnvelopeSpectrum(err_data, 'Fs', 1);
    title(['errEnvelopeSpectrum']);

    subFolder = fullfile(outputDir, datasetName, mfilename);

    figureName = sprintf("%s_%s_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, ENoB, verbose);
    saveVariable(subFolder, SNDR, verbose);
    saveVariable(subFolder, SFDR, verbose);
    saveVariable(subFolder, SNR, verbose);
    saveVariable(subFolder, THD, verbose);
    saveVariable(subFolder, pwr, verbose);
    saveVariable(subFolder, NF, verbose);
end
