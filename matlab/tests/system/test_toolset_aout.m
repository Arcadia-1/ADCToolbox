%% test_toolset_aout.m - Run AOUT toolset on analog ADC data
close all; clc; clear; warning("off")

%% Configuration
verbose = 0;
inputDir = "dataset";
outputDir = "test_output";
filesList = {"sinewave_amplitude_noise_0P001.csv"};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv', 'batch_sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% ADC Configuration
Resolution = 11;  % ADC resolution in bits

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);

    aout_data = readmatrix(dataFilePath);
    status = toolset_aout(aout_data, subFolder, 'Visible', verbose, 'Resolution', Resolution);
end
