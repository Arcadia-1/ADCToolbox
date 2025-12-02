%% test_FGCalSine.m - Unit test for FGCalSine function
close all; clc; clear;
%% Configuration
verbose = 0;
inputDir = "dataset/dout";
outputDir = "test_output";

filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'dout_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);


    read_data = readmatrix(dataFilePath);
    [N, M] = size(read_data);

    nomWeight = 2.^(M-1:-1:0);
    preCal = read_data * nomWeight';

    [weight, offset, postCal, ideal, err, freqCal] = FGCalSine(read_data, 'freq', 0, 'order', 5);

    subFolder = fullfile(outputDir, datasetName, mfilename);
    if ~isfolder(subFolder), mkdir(subFolder); end
    
    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [ENoB_pre, SNDR_pre, SFDR_pre, SNR_pre, THD_pre, pwr_pre, NF_pre, ~] = ...
        specPlot(preCal, 'label', 1, 'harmonic', 5, 'OSR', 1, 'NFMethod', 0);
    title(['Spectrum Before Calibration: ', datasetName], 'Interpreter', 'none');

    saveFig(subFolder, "specPlot_preCal_matlab.png", verbose);


    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [ENoB_post, SNDR_post, SFDR_post, SNR_post, THD_post, pwr_post, NF_post, ~] = ...
        specPlot(postCal, 'label', 1, 'harmonic', 5, 'OSR', 1, 'NFMethod', 0);
    title(['Spectrum After Calibration: ', datasetName], 'Interpreter', 'none');

    saveFig(subFolder, "specPlot_postCal_matlab.png", verbose);

    saveVariable(subFolder, weight, verbose);
    saveVariable(subFolder, offset, verbose);
    saveVariable(subFolder, postCal, verbose);
    saveVariable(subFolder, ideal, verbose);
    saveVariable(subFolder, err, verbose);
    saveVariable(subFolder, freqCal, verbose);

    saveVariable(subFolder, ENoB_pre, verbose);
    saveVariable(subFolder, ENoB_post, verbose);
end
