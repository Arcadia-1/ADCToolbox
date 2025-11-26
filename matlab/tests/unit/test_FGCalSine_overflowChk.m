%% test_FGCalSine_overflowChk.m - Unit test for overflowChk function
close all; clc; clear;
warning("off");

%% Configuration
verbose = 0;
inputDir = "dataset";
outputDir = "test_output";
filesList = autoSearchFiles({}, inputDir, 'dout_SAR_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('\n[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    [~, datasetName, ~] = fileparts(currentFilename);
    titleString = replace(datasetName, '_', '\_');

    weights_cal = FGCalSine(read_data);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    data_decom = overflowChk(read_data, weights_cal);
    title(titleString);

    subFolder = fullfile(outputDir, datasetName, mfilename);

    saveFig(subFolder, "overflowChk_matlab.png", verbose);

    [N, M] = size(data_decom);
    varNames = arrayfun(@(x) sprintf('bit_%d', x), M-1:-1:0, 'UniformOutput', false);
    dataDecomTable = array2table(data_decom, 'VariableNames', varNames);
    dataDecomPath = fullfile(subFolder, 'data_decom_matlab.csv');
    writetable(dataDecomTable, dataDecomPath);
end
