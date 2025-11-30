%% test_errPDF.m - Unit test for errPDF function
close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = "dataset/aout";
outputDir = "test_output";
filesList = autoSearchFiles({}, inputDir, 'sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);
    [~, datasetName, ~] = fileparts(currentFilename);

    read_data = readmatrix(dataFilePath);
    [data_fit, ~, ~, ~, ~] = sineFit(read_data);
    err_data = read_data - data_fit;

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [~, mu, sigma, KL_divergence, x, fx, gauss_pdf] = errPDF(err_data, ...
        'Resolution', 8, 'FullScale', max(read_data) - min(read_data));
    title(['errPDF: ', datasetName], 'Interpreter','none');
    set(gca, "FontSize", 16);

    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveFig(subFolder, "errPDF_matlab.png", verbose);
    saveVariable(subFolder, mu, verbose);
    saveVariable(subFolder, sigma, verbose);
    saveVariable(subFolder, KL_divergence, verbose);
    saveVariable(subFolder, x, verbose);
    saveVariable(subFolder, fx, verbose);
    saveVariable(subFolder, gauss_pdf, verbose);
end
