%% test_FGCalSine.m - Unit test for FGCalSine function
close all; clc; clear;
%% Configuration
verbose = 0;
inputDir = "dataset";
outputDir = "test_output";
filesList = {}; % Test data - leave empty to auto-search
filesList = autoSearchFiles(filesList, inputDir, 'dout_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('\n[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);


    read_data = readmatrix(dataFilePath);
    [weight, offset, postCal, ideal, err, freqCal] = FGCalSine(read_data, 'freq', 0, 'order', 5);


    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);
    if ~isfolder(subFolder), mkdir(subFolder); end

    saveVariable(subFolder, weight, verbose);
    saveVariable(subFolder, offset, verbose);
    saveVariable(subFolder, postCal, verbose);
    saveVariable(subFolder, ideal, verbose);
    saveVariable(subFolder, err, verbose);
    saveVariable(subFolder, freqCal, verbose);
end
