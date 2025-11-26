%% test_errHistSine.m - Unit test for errHistSine function
close all; clc; clear; warning("off")
%% Configuration
inputDir = "dataset/static_nonlin";
outputDir = "test_output/static_nonlin";

filesList = autoSearchFiles({}, inputDir, 'sinewave_*.csv');
if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
fprintf('=== test_errHistSine.m ===\n');
fprintf('[Testing] %d datasets...\n\n', length(filesList));

for k = 1:length(filesList)
    filename = filesList{k};
    filepath = fullfile(inputDir, filename);
    fprintf('[%d/%d] %s\n', k, length(filesList), filename);
    data = readmatrix(filepath);
    [~, datasetName, ~] = fileparts(filename);
    outFolder = fullfile(outputDir, datasetName, 'test_errHistSine');
    if ~isfolder(outFolder), mkdir(outFolder); end
    %% Phase mode
    figure('Position', [100, 100, 800, 600]);
    [~, freq, ~, ~, ~] = sineFit(data);
    [emean, erms, phase_code, anoi, pnoi] = errHistSine(data, 'bin', 360, 'fin', freq, 'disp', 1, 'mode', 0);
    sgtitle(replace(datasetName, '_', '\_'), 'Interpreter', 'tex');
    set(findall(gcf, 'Type', 'axes'), 'FontSize', 14);

    saveas(gcf, fullfile(outFolder, 'errHistSine_phase_matlab.png'));
    writetable(table(anoi, pnoi, 'VariableNames', {'anoi', 'pnoi'}), ...
        fullfile(outFolder, 'metrics_matlab.csv'));
    writetable(table(phase_code', emean', erms', 'VariableNames', {'phase_code', 'emean', 'erms'}), ...
        fullfile(outFolder, 'phase_histogram_matlab.csv'));
    %% Code mode
    figure('Position', [100, 100, 800, 600]);
    [emean_code, erms_code, code_axis, ~, ~, ~, ~, ~, k1, k2, k3] = ...
        errHistSine(data, 'bin', 256, 'fin', freq, 'disp', 1, 'mode', 1, 'polyorder', 3);
    sgtitle(replace(datasetName, '_', '\_'), 'Interpreter', 'tex');
    set(findall(gcf, 'Type', 'axes'), 'FontSize', 14);

    saveas(gcf, fullfile(outFolder, 'errHistSine_code_matlab.png'));
    writetable(table(code_axis', emean_code', erms_code', 'VariableNames', {'code', 'emean', 'erms'}), ...
        fullfile(outFolder, 'code_histogram_matlab.csv'));

    fprintf('  anoi=%.6f, pnoi=%.6f rad, k1=%.6f, k2=%.6f, k3=%.6f\n\n', anoi, pnoi, k1, k2, k3);
end
