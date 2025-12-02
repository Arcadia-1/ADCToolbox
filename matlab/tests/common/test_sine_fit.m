close all; clc; clear;

%% Configuration
verbose = 0;
inputDir = "reference_dataset/sinewave";
outputDir = "reference_output";
figureDir = "test_plots";

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

    % Perform sine fitting
    [data_fit, freq, mag, dc, phi] = sineFit(read_data);

    % Save variables
    subFolder = fullfile(outputDir, datasetName, mfilename);
    saveVariable(subFolder, freq, verbose);
    saveVariable(subFolder, mag, verbose);
    saveVariable(subFolder, dc, verbose);
    saveVariable(subFolder, phi, verbose);
    saveVariable(subFolder, data_fit, verbose);

    % Plotting Logic
    if freq > 0
        period_samples = round(1.0 / freq);
    else
        period_samples = length(read_data);
    end
    n_plot = min(max(period_samples, 20), length(read_data));

    figure('Position', [100, 100, 800, 600], "Visible", verbose);

    t_data = (0:n_plot-1);
    plot(t_data, read_data(1:n_plot), 'bo-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Original');
    hold on;

    t_dense = linspace(0, n_plot - 1, n_plot * 50);
    fitted_sine = mag * cos(2 * pi * freq * t_dense + phi) + dc;
    plot(t_dense, fitted_sine, 'r--', 'LineWidth', 2, 'DisplayName', 'Fitted Sine');

    xlabel('Sample Index');
    ylabel('Amplitude');
    title('sineFit: Original Data vs Fitted Sine');
    legend('Location', 'best');
    grid on;
    hold off;

    % Save figure
    figureName = sprintf("%s_%s_matlab.png", datasetName, mfilename);
    saveFig(figureDir, figureName, verbose);
end
