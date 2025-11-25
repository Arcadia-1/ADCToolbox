%% test_errPDF.m - Unit test for errPDF function
% Tests the errPDF function with sinewave error data
%
% Output structure:
%   test_output/<data_set_name>/test_errPDF/
%       metrics_matlab.csv      - mu, sigma, KL_divergence
%       pdf_data_matlab.csv     - x, fx, gauss_pdf
%       errPDF_matlab.png       - PDF plot

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
fprintf('=== test_errPDF.m ===\n');
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
    subFolder = fullfile(outputDir, datasetName, 'test_errPDF');
    if ~isfolder(subFolder)
        mkdir(subFolder);
    end

    %% Compute error data using sineFit
    [data_fit, ~, ~, ~, ~] = sineFit(read_data);
    err_data = read_data - data_fit;

    %% Run errPDF
    figure('Visible', 'off');
    [~, mu, sigma, KL_divergence, x, fx, gauss_pdf] = errPDF(err_data, ...
        'Resolution', 12, 'FullScale', max(read_data) - min(read_data));
    title(['errPDF: ', titleString]);

    % Save plot
    plotPath = fullfile(subFolder, 'errPDF_matlab.png');
    saveas(gcf, plotPath);
    fprintf('  [Saved] %s\n', plotPath);
    close(gcf);

    % Save metrics to CSV
    metricsTable = table(mu, sigma, KL_divergence, ...
        'VariableNames', {'mu', 'sigma', 'KL_divergence'});
    metricsPath = fullfile(subFolder, 'metrics_matlab.csv');
    writetable(metricsTable, metricsPath);
    fprintf('  [Saved] %s\n', metricsPath);

    % Save PDF data to CSV
    pdfTable = table(x', fx', gauss_pdf', ...
        'VariableNames', {'x', 'fx', 'gauss_pdf'});
    pdfPath = fullfile(subFolder, 'pdf_data_matlab.csv');
    writetable(pdfTable, pdfPath);
    fprintf('  [Saved] %s\n', pdfPath);

    fprintf('  [Results] mu=%.4f, sigma=%.4f, KL=%.6f\n\n', mu, sigma, KL_divergence);
end

fprintf('[test_errPDF COMPLETE]\n');
