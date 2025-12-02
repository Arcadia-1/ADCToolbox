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

    % Scale data by 2^Resolution
    scaled_data = read_data * (2 ^ Resolution);
    expected_max = 2 ^ Resolution;

    % Calculate INL/DNL
    [INL, DNL, code] = INLsine(scaled_data);

    % Calculate ranges for plot titles
    max_inl = max(INL);
    min_inl = min(INL);
    max_dnl = max(DNL);
    min_dnl = min(DNL);

    % Generate Plot
    figure('Position', [100, 100, 1000, 800], "Visible", verbose);

    % Top subplot: INL
    subplot(2, 1, 1);
    scatter(code, INL, 8, 'filled', 'MarkerFaceAlpha', 0.6);
    xlabel('Code');
    ylabel('INL (LSB)');
    grid on;
    title(sprintf('INL = [%.2f, %+.2f] LSB', min_inl, max_inl));
    ylim_min = min(min_inl, -1);
    ylim_max = max(max_inl, 1);
    ylim([ylim_min, ylim_max]);
    xlim([0, expected_max]);

    % Bottom subplot: DNL
    subplot(2, 1, 2);
    scatter(code, DNL, 8, 'filled', 'MarkerFaceAlpha', 0.6);
    xlabel('Code');
    ylabel('DNL (LSB)');
    grid on;
    title(sprintf('DNL = [%.2f, %.2f] LSB', min_dnl, max_dnl));
    ylim_min = min(min_dnl, -1);
    ylim_max = max(max_dnl, 1);
    ylim([ylim_min, ylim_max]);
    xlim([0, expected_max]);

    % Save figure
    subFolder = fullfile(outputDir, datasetName, mfilename);
    figureName = sprintf("%s_%s_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName, verbose);
end
