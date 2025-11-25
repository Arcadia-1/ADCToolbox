%% test_tomDecomp.m - Unit test for tomDecomp function
% Tests the Thompson Decomposition function with sinewave datasets
%
% Output structure:
%   test_output/<data_set_name>/test_tomDecomp/
%       decomp_data_matlab.csv  - signal, error, indep, dep (first 1000 samples)
%       metrics_matlab.csv      - phi, rms_error, rms_indep, rms_dep
%       tomDecomp_matlab.png

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
fprintf('=== test_tomDecomp.m ===\n');
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
    subFolder = fullfile(outputDir, datasetName, 'test_tomDecomp');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Find input frequency
    re_fin = findFin(read_data);

    %% Run tomDecomp
    figure('Visible', 'off');
    [signal, error, indep, dep, phi] = tomDecomp(read_data, re_fin, 10, 1);
    sgtitle(['tomDecomp: ', titleString]);

    % Save plot
    plotPath = fullfile(subFolder, 'tomDecomp_matlab.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);
    close(gcf);

    % Save decomposition data (first 1000 samples for comparison)
    N_save = min(1000, length(signal));
    decompTable = table(signal(1:N_save), error(1:N_save), indep(1:N_save), dep(1:N_save), ...
        'VariableNames', {'signal', 'error', 'indep', 'dep'});
    decompPath = fullfile(subFolder, 'decomp_data_matlab.csv');
    writetable(decompTable, decompPath);
    fprintf('  [Saved] %s\n', decompPath);

    % Save metrics
    rms_error = rms(error);
    rms_indep = rms(indep);
    rms_dep = rms(dep);
    metricsTable = table(phi, rms_error, rms_indep, rms_dep, ...
        'VariableNames', {'phi', 'rms_error', 'rms_indep', 'rms_dep'});
    metricsPath = fullfile(subFolder, 'metrics_matlab.csv');
    writetable(metricsTable, metricsPath);
    fprintf('  [Saved] %s\n', metricsPath);

    fprintf('  [Results] phi=%.4f rad, rms_error=%.6f, rms_indep=%.6f, rms_dep=%.6f\n\n', ...
        phi, rms_error, rms_indep, rms_dep);
end

fprintf('[test_tomDecomp COMPLETE]\n');
