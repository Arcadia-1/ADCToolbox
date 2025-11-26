%% test_INLsine.m - Unit test for INLsine function
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

    read_data = readmatrix(dataFilePath);

    resolution = 10;
    [INL, DNL, code] = INLsine(read_data*2^resolution, 0.01);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    subplot(2, 1, 1);
    scatter(code, DNL, '.');
    grid on;
    xlabel('Code');
    ylabel('DNL (LSB)');
    xlim([0, 2^resolution])
    ylim([min(DNL) - 0.5, max(DNL) + 0.5])
    text(0.02, 0.95, sprintf('DNL = [%0.2f, %0.2f] LSB', max(DNL), min(DNL)), ...
        'Units', 'normalized', 'FontSize', 16, 'VerticalAlignment', 'top');
    set(gca, "FontSize", 16);

    subplot(2, 1, 2);
    plot(code, INL, 'b-', 'LineWidth', 1);
    grid on;
    xlabel('Code');
    ylabel('INL (LSB)');
    xlim([0, 2^resolution])
    ylim([min(INL) - 0.5, max(INL) + 0.5])
    text(0.02, 0.95, sprintf('INL = [%0.2f, %0.2f] LSB', max(INL), min(INL)), ...
        'Units', 'normalized', 'FontSize', 16, 'VerticalAlignment', 'top');
    set(gca, "FontSize", 16);


    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveFig(subFolder, "INLsine_matlab.png", verbose);
    saveVariable(subFolder, code, verbose);
    saveVariable(subFolder, DNL, verbose);
    saveVariable(subFolder, INL, verbose);
end
