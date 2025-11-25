%% test_errHistSine.m - Unit test for errHistSine function
% Tests the errHistSine function with sinewave datasets
%
% Output structure:
%   test_output/<data_set_name>/test_errHistSine/
%       metrics_matlab.csv          - anoi, pnoi
%       phase_histogram_matlab.csv  - phase_code, emean, erms
%       errHistSine_phase_matlab.png
%       errHistSine_code_matlab.png

close all; clc; clear; warning("off")

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
fprintf('=== test_errHistSine.m ===\n');
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
    subFolder = fullfile(outputDir, datasetName, 'test_errHistSine');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Run errHistSine - Phase mode (mode=0)
    figure('Visible', 'off');


    [~, freq, ~, ~, ~] = sineFit(read_data);
    [emean, erms, phase_code, anoi, pnoi, ~, ~] = errHistSine(read_data, 'bin',360, 'fin',freq, 'disp',1, 'mode',0);
    sgtitle(['errHistSine (phase): ', titleString]);

    % Save plot
    plotPath = fullfile(subFolder, 'errHistSine_phase_matlab.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);
    close(gcf);

    % Save metrics
    metricsTable = table(anoi, pnoi, ...
        'VariableNames', {'anoi', 'pnoi'});
    metricsPath = fullfile(subFolder, 'metrics_matlab.csv');
    writetable(metricsTable, metricsPath);
    fprintf('  [Saved] %s\n', metricsPath);

    % Save histogram data
    histTable = table(phase_code', emean', erms', ...
        'VariableNames', {'phase_code', 'emean', 'erms'});
    histPath = fullfile(subFolder, 'phase_histogram_matlab.csv');
    writetable(histTable, histPath);
    fprintf('  [Saved] %s\n', histPath);

    %% Run errHistSine - Code mode (mode=1)
    figure('Visible', 'off');
    [emean_code, erms_code, code_axis, ~, ~, ~, ~] = errHistSine(read_data, 'bin',256, 'fin',freq, 'disp',1, 'mode',1);
    sgtitle(['errHistSine (code): ', titleString]);

    % Save plot
    plotPath = fullfile(subFolder, 'errHistSine_code_matlab.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);
    close(gcf);

    % Save code histogram data
    codeHistTable = table(code_axis', emean_code', erms_code', ...
        'VariableNames', {'code', 'emean', 'erms'});
    codeHistPath = fullfile(subFolder, 'code_histogram_matlab.csv');
    writetable(codeHistTable, codeHistPath);
    fprintf('  [Saved] %s\n', codeHistPath);

    fprintf('  [Results] anoi=%.6f, pnoi=%.6f rad\n\n', anoi, pnoi);
end

fprintf('[test_errHistSine COMPLETE]\n');
