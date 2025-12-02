close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = "reference_dataset/sinewave";
outputDir = "reference_output";
figureDir = "test_plots";
Resolution = 12;

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

    % Compute error data and full scale
    [data_fit, ~, ~, ~, ~] = sineFit(read_data);
    err_data = read_data - data_fit;
    FullScale = max(read_data) - min(read_data);

    % Run errPDF
    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [noise_lsb, mu, sigma, KL_divergence, x, fx, gauss_pdf] = errPDF(err_data, 'Resolution', Resolution, 'FullScale', FullScale);

    title(mfilename);

    % Save outputs
    subFolder = fullfile(outputDir, datasetName, mfilename);
    figureName = sprintf("%s_%s_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName, verbose);
    saveVariable(subFolder, noise_lsb, verbose);
    saveVariable(subFolder, mu, verbose);
    saveVariable(subFolder, sigma, verbose);
    saveVariable(subFolder, KL_divergence, verbose);
    saveVariable(subFolder, x, verbose);
    saveVariable(subFolder, fx, verbose);
    saveVariable(subFolder, gauss_pdf, verbose);
end
