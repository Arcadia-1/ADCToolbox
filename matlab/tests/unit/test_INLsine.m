%% test_INLsine.m - Unit test for INLsine function
% Tests the INLsine function to compute INL/DNL from sinewave data
%
% Output structure:
%   test_output/<data_set_name>/test_INLsine/
%       inl_dnl_matlab.csv      - code, INL, DNL
%       metrics_matlab.csv      - max_INL, min_INL, max_DNL, min_DNL
%       INLsine_matlab.png

close all; clc; clear;

%% Configuration
inputDir = "dataset";
outputDir = "test_output";

% Test datasets - leave empty to auto-search
filesList = {};
filesList = autoSearchFiles(filesList, inputDir, 'sinewave_*.csv');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Test Loop
fprintf('=== test_INLsine.m ===\n');
fprintf('[Testing] %d datasets...\n\n', length(filesList));

for k = 1:length(filesList)
    currentFilename = filesList{k};
    dataFilePath = fullfile(inputDir, currentFilename);

    if ~isfile(dataFilePath)
        fprintf('[%d/%d] %s - NOT FOUND, skipping\n\n', k, length(filesList), currentFilename);
        continue;
    end
    fprintf('[%d/%d] [Processing] %s\n', k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    % Extract dataset name
    [~, datasetName, ~] = fileparts(currentFilename);
    titleString = replace(datasetName, '_', '\_');

    % Create output subfolder
    subFolder = fullfile(outputDir, datasetName, 'test_INLsine');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Run INLsine
    resolution = 12;
    [INL, DNL, code] = INLsine(read_data*2^resolution, 0.01);

    %% Plot
    figure('Visible', 'off');
    subplot(2,1,1);
    bar(code, DNL);
    grid on;
    xlabel('Code');
    ylabel('DNL (LSB)');
    title(['DNL: ', titleString]);
    xlim([0, 2^resolution])

    subplot(2,1,2);
    plot(code, INL, 'b-', 'LineWidth', 1);
    grid on;
    xlabel('Code');
    ylabel('INL (LSB)');
    title(['INL: ', titleString]);
    xlim([0, 2^resolution])


    % Save plot
    plotPath = fullfile(subFolder, 'INLsine_matlab.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);
    close(gcf);

    % Save INL/DNL data to CSV
    inlDnlTable = table(code', DNL', INL', ...
        'VariableNames', {'code', 'DNL', 'INL'});
    inlDnlPath = fullfile(subFolder, 'inl_dnl_matlab.csv');
    writetable(inlDnlTable, inlDnlPath);
    fprintf('  [Saved] %s\n', inlDnlPath);
    % Save metrics
    max_DNL = max(DNL);
    min_DNL = min(DNL);
    max_INL = max(INL);
    min_INL = min(INL);
    metricsTable = table(max_DNL, min_DNL, max_INL, min_INL, ...
        'VariableNames', {'max_DNL', 'min_DNL', 'max_INL', 'min_INL'});
    metricsPath = fullfile(subFolder, 'metrics_matlab.csv');
    writetable(metricsTable, metricsPath);
    fprintf('  [Saved] %s\n', metricsPath);
    fprintf('  [Results] DNL: [%.2f, %.2f], INL: [%.2f, %.2f] LSB\n\n', ...
        min_DNL, max_DNL, min_INL, max_INL);
end

fprintf('[test_INLsine COMPLETE]\n');
