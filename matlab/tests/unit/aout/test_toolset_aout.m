%% test_toolset_aout.m - Run AOUT toolset on analog ADC data
close all; clc; clear; warning("off")

%% Configuration
verbose = 0;
inputDir = "dataset/aout/sinewave";
outputDir = "test_output";

filesList = {"sinewave_noise_270uV"};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv');
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

    % Run toolset_aout to generate 9 individual plots
    aout_data = readmatrix(dataFilePath);
    status = toolset_aout(aout_data, subFolder, 'Visible', verbose, 'Resolution', Resolution);

    % Generate panel (3x3 overview of all 9 plots)
    if status.success
        panel_status = toolset_aout_panel(status.plot_files, subFolder, 'Visible', verbose, 'Prefix', 'aout');
    else
        fprintf('[WARNING] toolset_aout failed, skipping panel generation\n');
    end
end
