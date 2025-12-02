%% test_FGCalSine_overflowChk.m - Unit test for overflowChk function
close all; clc; clear;
warning("off");

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
    titleString = replace(datasetName, '_', '\_');

    read_data = readmatrix(dataFilePath);
    [weights_cal, offset, postCal, ideal, err, freqCal] = FGCalSine(read_data);

    ENoB = specPlot(postCal, "isplot", 0);
    fprintf("[%s] [ENoB = %0.2f bits]\n", mfilename, ENoB);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    data_decom = overflowChk(read_data, weights_cal);
    title(['overflowChk: ', titleString]);
    set(gca, "FontSize", 16);

    subFolder = fullfile(outputDir, datasetName, mfilename);

    saveFig(subFolder, "overflowChk_matlab.png", verbose);
    saveVariable(subFolder, data_decom, verbose);
end
