%% test_toolset_dout.m - Run DOUT toolset on ADC digital output
close all; clc; clear; warning("off")

%% Configuration
verbose = 0;
inputDir = "dataset/dout";
outputDir = "test_output";
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'dout_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Calibration Configuration
Order = 5;  % Polynomial order for FGCalSine

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);
    
    bits = readmatrix(dataFilePath);
    status = toolset_dout(bits, subFolder, 'Visible', verbose, 'Order', Order);
end