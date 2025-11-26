%% test_errHistSine_phase.m - Unit test for errHistSine phase mode
close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = "dataset";
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
    [~, freq, ~, ~, ~] = sineFit(read_data);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [emean, erms, phase_code, anoi, pnoi] = errHistSine(read_data, 'bin', 360, 'fin', freq, 'disp', 1, 'mode', 0);
    sgtitle(replace(datasetName, '_', '\_'), 'Interpreter', 'tex');
    set(findall(gcf, 'Type', 'axes'), 'FontSize', 14);

    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveFig(subFolder, "errHistSine_phase_matlab.png", verbose);
    saveVariable(subFolder, anoi, verbose);
    saveVariable(subFolder, pnoi, verbose);
    saveVariable(subFolder, phase_code, verbose);
    saveVariable(subFolder, emean, verbose);
    saveVariable(subFolder, erms, verbose);
end
