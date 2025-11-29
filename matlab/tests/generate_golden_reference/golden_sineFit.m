%% golden_sineFit.m - Golden reference test for sineFit
%
% This test processes only the datasets listed in golden_data_list.txt
% Use this to generate golden references, not for comprehensive testing.

close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = "dataset";
outputDir = "test_reference";

% Read golden data list
goldenListFile = "test_reference/golden_data_list.txt";
if ~isfile(goldenListFile)
    error('Golden data list not found: %s', goldenListFile);
end

fid = fopen(goldenListFile, 'r');
filesList = {};
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && ~isempty(line) && ~startsWith(strtrim(line), '#')
        filesList{end+1} = strtrim(line);
    end
end
fclose(fid);

fprintf('Golden test: processing %d files from golden_data_list.txt\n\n', length(filesList));

if ~isfolder(outputDir), mkdir(outputDir); end

%% Test Loop
for k = 1:length(filesList)
    currentFilename = filesList{k};
    filepath = fullfile(inputDir, currentFilename);
    fprintf('[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(filepath);
    [data_fit, freq, mag, dc, phi] = sineFit(read_data);

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    period = round(1/freq);
    n_samples = min(max(period, 20), length(read_data));
    t = 1:n_samples;

    plot(t, read_data(1:n_samples), '-o', 'LineWidth', 2, 'DisplayName', 'Original Data');
    hold on;

    t_dense = linspace(1, n_samples, n_samples*100);
    fitted_sine = mag * cos(2*pi*freq*(t_dense - 1)+phi) + dc;
    plot(t_dense, fitted_sine, '--', 'LineWidth', 1, 'DisplayName', 'Fitted Sine');

    set(gca, "FontSize",16)
    xlabel('Sample');
    ylabel('Amplitude');
    legend('Location', 'northwest');
    grid on;
    ylim([min(fitted_sine) - 0.1, max(fitted_sine) + 0.2])

    [~, datasetName, ~] = fileparts(currentFilename);
    subFolder = fullfile(outputDir, datasetName, "test_sineFit");
    saveFig(subFolder, "sineFit_matlab.png", verbose);
    saveVariable(subFolder, freq, verbose);
    saveVariable(subFolder, mag, verbose);
    saveVariable(subFolder, dc, verbose);
    saveVariable(subFolder, phi, verbose);
    saveVariable(subFolder, data_fit, verbose);
end
