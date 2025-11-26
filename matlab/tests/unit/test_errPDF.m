%% test_errPDF.m - Unit test for errPDF function
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
    fprintf('\n[%s] [%d/%d] [%s]\n', mfilename, k, length(filesList), currentFilename);

    read_data = readmatrix(dataFilePath);

    [~, datasetName, ~] = fileparts(currentFilename);
    titleString = replace(datasetName, '_', '\_');

    [data_fit, ~, ~, ~, ~] = sineFit(read_data);
    err_data = read_data - data_fit;

    figure('Position', [100, 100, 800, 600], "Visible", verbose);
    [~, mu, sigma, KL_divergence, x, fx, gauss_pdf] = errPDF(err_data, ...
        'Resolution', 12, 'FullScale', max(read_data) - min(read_data));
    title(['errPDF: ', titleString]);

    subFolder = fullfile(outputDir, datasetName, mfilename);

    saveFig(subFolder, "errPDF_matlab.png", verbose);

    metricsTable = table(mu, sigma, KL_divergence, ...
        'VariableNames', {'mu', 'sigma', 'KL_divergence'});
    metricsPath = fullfile(subFolder, 'metrics_matlab.csv');
    writetable(metricsTable, metricsPath);

    pdfTable = table(x', fx', gauss_pdf', ...
        'VariableNames', {'x', 'fx', 'gauss_pdf'});
    pdfPath = fullfile(subFolder, 'pdf_data_matlab.csv');
    writetable(pdfTable, pdfPath);
end
